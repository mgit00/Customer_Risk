import torch 

from torch.nn import CrossEntropyLoss, MSELoss # MSELoss for continuous loss 
from torch.optim import Adam 

from models import FNN 
class Train: 
    '''
    implements training loop 
    called from model_helper 
    calls models.py (depending on the model selected)
    '''
    def __init__(self, args): 
        self.args= args 

    def training_loop(self, train_test_dataloaders):
        '''
        returns model state

        since cross entropy is used, no need for softmax layer in model 
            cross entropy= softmax + NLL 
        
        result shapes for gradient update: 
        - model output: (batch size, number of classes ) [2d]
        - target shape: (batch size) [1d] # integers, which refer to the proper class 
        '''

        train_loader, test_loader = train_test_dataloaders 

        # ensure that I use gpu if available 
        self.args.device= torch.device('cuda' if torch.cuda.is_available() else 'cpu') # added into args 
        print('device used: ', self.args.device)
        # x, y = x.to(self.args.device), y.to(self.args.device) # this is to be done when iterating dataloader 

        # layer sizes 
        input_size= 35 # next(iter(train_loader))...... 
        hidden_sizes = 20 
        output_size = 1 # number of classes in risk_level 


        # get the model, optimizer, and loss function 
        model = FNN(self.args, input_size, hidden_sizes, output_size)
        # place model onto gpu 
        model = model.to(self.args.device)

        loss= MSELoss() # CrossEntropyLoss() # combines softmax and NLL 
        optimizer= Adam(model.parameters(), lr= self.args.lr )

        print('model is: ', model)

        print('the dtypes for x data and model are: ', 
              [next(iter(train_loader))[0].dtype] ,
              {param.dtype for param in model.parameters()}) # pytorch always initalized to float32 
        print('the dtypes for the target data are: ', next(iter(train_loader))[1].dtype) # dtype long for CELoss 

        


        for epoch in range(self.args.epochs): 
            print('epoch', epoch)
            epoch_loss = 0 
            epoch_loss_test = 0 

            model.train()
            for x, y, in train_loader: 
                # place x,y onto gpu 
                x= x.to(self.args.device)
                y= y.to(self.args.device)


                pred  = model(x).flatten()
                batch_loss = loss(pred, y) # y is dtype long for CE Loss 

                epoch_loss += batch_loss.item()
                
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()


            model.eval()
            for x,y in test_loader: 
                # place x,y onto gpu 
                x= x.to(self.args.device)
                y= y.to(self.args.device)

                pred = model(x).flatten()
                new = torch.argmax(pred, axis= -1 )
               
                batch_loss = loss(pred, y)

                epoch_loss_test+= batch_loss.item()


            print('epoch', epoch, 'losses for train and test: ', epoch_loss, epoch_loss_test)

        # raise Exception('here')
        return None 
        