from typeguard import typechecked 
from model_metrics import ModelMetrics

import logging 

class ModelState: 
    '''
    this class will keep the model state of the model being trained
    this will make it easier to save and load in the model 
    the class is called by Train.train() from train.py

    model state will be: 
    - args
    - epochs (int)
    - state dict of model 
    - state dict of optimizer 
    - loss function used 
    - list of model metrics: 
        - losses 
        - test losses 
        if cross entropy loss used: 
            - accuracy 
            - test accuracy 
            - f1 
            - test f1 
    '''

    def __init__(self, args): 
        self.args = args # note args are dynamic and will be adjusted through pipeline
        # args saved are the args called at the time of save model

        self.logger = logging.getLogger(self.args.logger_name)

        self.model_state_dict= None 
        self.optimizer_state_dict= None 

        self.epochs = 0 
        self.loss_func_mode= None 

        self.losses= [] 
        self.testlosses= [] 
        self.accuracy= [] 
        self.testaccuracy= [] 
        self.f1= [] 
        self.testf1= [] 

        

    @typechecked
    def preserve_model_state(self,  model, optimizer, modelmetrics: ModelMetrics ): 
        '''
        saves and returns current model state 
        '''
        self.logger.info('saving/updating model state')

        self.args = self.args 

        self.model_state_dict= model.state_dict()
        self.optimizer_state_dict= optimizer.state_dict()

        self.epochs += len(modelmetrics.losses) ##### have to deal with this... 
        self.loss_func_mode= modelmetrics.mode # string, not instance of LossFunc 

        self.losses.extend(modelmetrics.losses )
        self.testlosses.extend(modelmetrics.losses_test )

        if  modelmetrics.mode == 'cel': 
            # these attributes will exist in modelstate regardless, but they dont exist in modelmetrics
            self.accuracy.extend( modelmetrics.accuracy)
            self.testaccuracy.extend( modelmetrics.accuracy_test ) 
            self.f1.extend( modelmetrics.f1score ) 
            self.testf1.extend( modelmetrics.f1score_test) 

        return vars(self)

    def loading_model_state(self, model_state) -> dict: 
        '''
        loads in the model 
        model state should have right info in it to start with, guanteed with running preserve_model_state
        '''

        for key,value in model_state.items(): 
            setattr(self, key,value)
        
        self.logger.info('model loaded in')

        return vars(self)

