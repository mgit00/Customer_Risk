import torch 
import logging 
from models import SLP

from torch.optim import Adam 
from torch.nn import CrossEntropyLoss, MSELoss 

class Train: 
    def __init__(self, args) -> None:
        self.args = args 
        
        self.logger = logging.getLogger(self.args.logger_name)

    def get_loss_func(self): 
        if self.args.output_mode =='classification': 
            self.logger.info('loss function: cross entropy')
            return CrossEntropyLoss()
        else : 
            self.logger.info('loss function: mse')
            return MSELoss()
    
    def freeze_params(self, submodel): 
        '''
        this freezes the params of a model
        mainly for freezing submodel whose params shouldn't be updated anymore  
        used when needing to freeze the autoencoder layers during fnn forward()
        '''
        for param in submodel.parameters(): 
            param.requires_grad = False 

    def train(self, data: tuple, model) -> dict: # returns model_state  
        '''
        training loop 
        - initializes model, loss func, and optimizer 
        - places model and data on gpu if available 
        - training loop for epoch 
        - also deals with creating model state at end of training 
        - keeps track of model training/ model metrics 

        to keep track of model metrics: make new function 
        - loss tracker 
        - accuracy 
        - f1 
        '''
        self.logger.debug('in training loop ')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info('device used: ' + str(device) ) # torch.cuda.current_device() gives integer 

        train_dataloader, test_dataloader = data 
        # place data on device when iterating 


        # place model on device 
        model= model.to(device)
        # ensure that all params have gradient computations enabled 
            # this can be false for fnn of autoencoder kemans fnn model since autoencoder shouldn't be trained anymore
        self.logger.debug('all model params require gradient: '+ str(all(param.requires_grad for param in model.parameters())))
        
        # make optimizer 
        optimizer = Adam(model.parameters(), lr = self.args.lr)

        # get loss function CEL or MSEL based on classification or regression 
        loss_func = self.get_loss_func()

        

        losses=[]
        losses_test=[]

        # do training loop 
        for epoch in range(self.args.epochs): 
            epoch_loss = 0 
            epoch_loss_test = 0 

            model.train()
            # loop through batches from dataloader 
            for x,y in train_dataloader: 
                # place onto gpu if available 
                x,y = x.to(device), y.to(device)
                
                pred = model(x)
                loss = loss_func(pred, y)
                
                epoch_loss+= loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            model.eval() # disable dropout if in model and freeze batch normalization stats
            with torch.no_grad(): # disable gradient computation
                for x,y in test_dataloader: 
                    # place onto gpu if needed 
                    x,y = x.to(device), y.to(device)

                    pred = model(x)
                    loss = loss_func(pred,y)

                    epoch_loss_test+= loss.item()

        

            print('Epoch', epoch, 'Train Loss', epoch_loss, 'Test Loss', epoch_loss_test)
            losses.append(epoch_loss)
            losses_test.append(epoch_loss_test)
        


        model_state = {
            'args': self.args, 
            # 'epochs'; epoch count , 
            'losses': losses, 
            'losses_test' : losses_test, 
        }

        self.logger.debug('model training complete ')
        return model_state 