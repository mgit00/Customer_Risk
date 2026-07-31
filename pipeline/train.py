import torch 

from torch.nn import CrossEntropyLoss , MSELoss # for continuous loss 
from torch.optim import Adam 

from models import SLP, FNN, KMeansFNN, AutoencoderKMeansFNN 

class Train: 
    '''
    implements training loop 
    called from model_helper 
    calls models.py (depending on the model selected)
    '''
    def __init__(self, args, loaded_model= None ): 
        self.args= args 
        self.loaded_model= loaded_model

    def get_model(self, input_size, output_size): 
        ''' 
        this function gets the right model
        selected from the args line 
        '''
        
        model_type= self.args.model_type
        if model_type == 'slp': 
            model = SLP(input_size, output_size)
        elif model_type == 'fnn': 
            model = FNN(input_size, self.args.hidden_sizes, output_size)
        elif model_type == 'kmeans_fnn': 
            model= KMeansFNN(self.args.k_value, input_size, self.args.hidden_sizes, output_size)
        elif model_type== 'autoencoder_kmeans_fnn': 
            model = AutoencoderKMeansFNN(input_size, self.args.hidden_sizes, output_size)

        return model 
    


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
        hidden_sizes = self.args.hidden_sizes  
        if self.args.target_mode == 'classification': 
            output_size = 5 # number of classes in risk_level 
        else: 
            output_size = 1 


        # get the model, optimizer, and loss function 
        model = self.get_model(input_size, output_size) # hidden size from args  # FNN(self.args, input_size, hidden_sizes, output_size)
        print('model is: ', model)
        # place model onto gpu 
        model = model.to(self.args.device)

        if self.args.target_mode == 'classification': 
            loss= CrossEntropyLoss() # combines softmax and NLL 
        else: 
            loss= MSELoss()

        optimizer= Adam(model.parameters(), lr= self.args.lr )


        # tracking loss, epoch, metrics
        train_loss= []
        test_loss=[]
        curr_epoch= 0 



        # if the model is being loaded in, update the state_dicts, losses .... 
        if self.loaded_model: 
            model.load_state_dict( self.loaded_model['model_state_dict'])
            optimizer.load_state_dict(self.loaded_model['optimizer_state_dict'])
            train_loss=self.loaded_model['train_losses']
            test_loss=self.loaded_model['test_losses']
            curr_epoch = self.loaded_model['epochs_completed'] 
            print('model is loaded in ')


        
        print('the dtypes for x data and model are: ', 
              [next(iter(train_loader))[0].dtype] ,
              {param.dtype for param in model.parameters()}) # pytorch always initalized to float32 
        print('the dtypes for the target data are: ', next(iter(train_loader))[1].dtype) # dtype long for CELoss 

        
        for epoch in range(curr_epoch, curr_epoch+ self.args.epochs): 
            epoch_loss = 0 
            epoch_loss_test = 0 

            model.train()
            for x, y, in train_loader: 
                # place x,y onto gpu 
                x= x.to(self.args.device)
                y= y.to(self.args.device)


                pred  = model(x)
                if self.args.target_mode == 'regression': 
                    pred= pred.flatten()# for correct shape MSELoss 

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

                pred = model(x)
                if self.args.target_mode == 'regression': 
                    pred= pred.flatten()# for correct shape MSELoss 
                    
                new = torch.argmax(pred, axis= -1 )
               
                batch_loss = loss(pred, y)

                epoch_loss_test+= batch_loss.item()

            train_loss.append(epoch_loss)
            test_loss.append(epoch_loss_test)
            print('epoch', epoch, 'train losses: ', epoch_loss, 'test losses: ',epoch_loss_test)

        model_state= {
            'args': self.args, 
            'model_state_dict': model.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict(), 

            'epochs_completed': curr_epoch + self.args.epochs, # ... 
            'train_losses': train_loss, 
            'test_losses': test_loss 
        }
        return model_state 
        