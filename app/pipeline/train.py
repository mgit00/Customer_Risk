import torch 
import logging 
from .model_metrics import ModelMetrics, LossFunc, Levels
from .model_state import ModelState
from .utils import Utils 

from torch.optim import Adam 
from torch.nn import CrossEntropyLoss, MSELoss 

class Train: 
    def __init__(self, args) -> None:
        self.args = args 
        
        self.logger = logging.getLogger(self.args.logger_name)


    def get_loss_func(self, model_name ): 
        #  if the model is autoencoder, the loss is automatically MSE 
            # since the target is the input 
        if model_name =='Autoencoder': 
            self.logger.info('loss function autoencoder: mse')
            return MSELoss()
        elif self.args.output_mode =='classification': 
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

        to keep track of model metrics: make new function (ModelMetrics)
        this creates an interface to track the metrics 
            - loss tracker 
            - accuracy 
            - f1 
        '''

        self.logger.debug('in training loop ')

        self.args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info('device used: ' + str(self.args.device) ) # torch.cuda.current_device() gives integer 

        ''' data and model setting up '''
        self.logger.debug('setting up data, model, optimizer, loss function ')
        train_dataloader, test_dataloader = data 

        # place data on device when iterating, this places batch on device to save memory

        # place model on device 
        model= model.to(self.args.device)

        # ensure that all params have gradient computations enabled 
            # this can be false for fnn of autoencoder kemans fnn model since autoencoder shouldn't be trained anymore
        self.logger.debug('all model params require gradient: '+ str(all(param.requires_grad for param in model.parameters())))
        
        # make optimizer 
        optimizer = Adam(model.parameters(), lr = self.args.lr)

        # get loss function CEL or MSEL based on classification or regression 
        loss_func = self.get_loss_func(model_name= model.__class__.__name__)


        ''' model metrics and state are initialized ''' 
        self.logger.debug('initializing model metrics and model state class ')
        # initlalize model metrics, these are never affected by a model being loaded in 
            # these are specific to 1 call fo this function 
        mode = LossFunc.MSE if loss_func.__class__.__name__ == 'MSELoss' else LossFunc.CrossEntropyLoss
        modelmetrics= ModelMetrics(self.args, mode = mode )
        modelmetrics.initialize_metrics()

        # initialize model state 
        model_state_class = ModelState(self.args) # init to empty 

        # if needed, read in loaded model and optim state_dicts 
        if self.args.load_model: 
            self.logger.debug('loading in model from: ' + self.args.model_filename)
            loaded_model = Utils.load_model(self.args.model_filename)
            loaded_model_state = model_state_class.loading_model_state(loaded_model)
            # update state dicts 
            model.load_state_dict(model_state_class.model_state_dict)
            optimizer.load_state_dict(model_state_class.optimizer_state_dict)
            self.logger.info('model loaded in from: ' + self.args.model_filename)


        ''' train loop '''
        self.logger.debug('starting training loop')
        ### do training loop 
        for epoch in range(self.args.epochs): 
            # print(f"Epoch {epoch}")
            # print("Bias:", model.layer.bias.detach().cpu())
            # print("Weight norms:", model.layer.weight.norm(dim=1).detach().cpu())
            # print(loss_func)

            modelmetrics.initalize_metric_trackers() # reset metrics for this epoch 

            model.train()
            # loop through batches from dataloader 
            for x,y in train_dataloader: 
                # place onto gpu if available 
                x,y = x.to(self.args.device), y.to(self.args.device)
                
                pred = model(x)

                

                if mode == LossFunc.CrossEntropyLoss: # if loss_func.__class__.__name__ == 'CrossEntropyLoss': 
                    y= y.flatten() # for CEL
                loss = loss_func(pred, y) 
                
                modelmetrics.update_batch_level(loss.item(),  pred, y, level = Levels.TRAIN)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            model.eval() # disable dropout if in model and freeze batch normalization stats
            with torch.no_grad(): # disable gradient computation
                for x,y in test_dataloader: 
                    # place onto gpu if needed 
                    x,y = x.to(self.args.device), y.to(self.args.device)

                    pred = model(x)
                    # print(pred[:10])
                    
                    # fixme print(torch.unique(torch.argmax(pred, dim = 1), return_counts= True ))

                    if mode == LossFunc.CrossEntropyLoss: # if loss_func.__class__.__name__ == 'CrossEntropyLoss': 
                        y= y.flatten() # for CEL
                    loss = loss_func(pred,y)

                    modelmetrics.update_batch_level(loss.item(), pred, y, level = Levels.TEST)

        

            modelmetrics.update_epoch_level() # calculate the metrics at the end of this epoch 
            print('Epoch', model_state_class.epochs+ epoch, 'Train Loss', modelmetrics.epoch_loss, 'Test Loss', modelmetrics.epoch_loss_test)
            


        model_state = model_state_class.preserve_model_state(model, optimizer, modelmetrics)
        self.logger.info('model state creates/updated with training information') 

        self.logger.debug('model training complete ')
        return model_state 