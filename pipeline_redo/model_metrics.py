import logging 
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from enum import Enum 

from typeguard import typechecked 

class Levels(Enum): 
    '''
    used to choose argument choices
    '''
    TRAIN= 'train'
    TEST= 'test'
class LossFunc(Enum): 
    MSE= 'mse'
    CrossEntropyLoss= 'cel'


class ModelMetrics: 
    '''
    class to deal with creating and tracking model training metrics
    - all of this functionality can be kept in train, but this makes the code more flexible and easy to understand 
    - tracks various metrics 
    - initialized in Train.train <- this class is unique to each call of train()
        - does not cover info gained from loading in a model 

    metrics can change with output_mode classification or regression 

    given the loss, update the metrics 
        the loss can vary based on epoch loss vs train loss
        this means that a epoch_update and train_update has to be done

        from this, the model state info can be extracted 
    '''

    def __init__(self, args, mode): 
        self.args =args 
        # note args are self updating: changing args after initialization changes them here too 
            # namespace property
        self.mode = mode.value # mse or cel 
        self.logger = logging.getLogger(self.args.logger_name)

    ### epoch level 
    def initialize_metrics(self):
        '''
        initializes metrics for the entire training loop 
        '''
        self.losses = []
        self.losses_test = []

        if self.mode == 'mse': 
            return 
        
        self.accuracy= []
        self.accuracy_test = []

        self.f1score = []
        self.f1score_test =[]

        self.logger.debug('metrics are initialized')
        return 

    def update_epoch_level(self): 
        '''
        epoch level updates
        '''
        self.losses.append(self.epoch_loss)
        self.losses_test.append(self.epoch_loss_test)

        if self.mode == 'cel': 
            self.accuracy.append(self.accuracy_metric.compute())
            self.accuracy_test.append(self.accuracy_metric_test.compute())

            self.f1score.append(self.f1_metric.compute())
            self.f1score_test.append(self.f1_metric_test.compute())

        # reset in train loop (initalize_metric_trackers())
        return 

    ### dataloader level 
    def initalize_metric_trackers(self): 
        '''
        initialize classification metrics
            has to be done every epoch 
            done in train loop  
        '''
        self.epoch_loss = 0 
        self.epoch_loss_test = 0 

        if self.mode == 'mse': 
            return 

        self.f1_metric = MulticlassF1Score(num_classes= 5)
        self.f1_metric_test= MulticlassF1Score(num_classes= 5)
        self.accuracy_metric = MulticlassAccuracy(num_classes= 5 )
        self.accuracy_metric_test = MulticlassAccuracy(num_classes= 5 )
        
        self.f1_metric.to(self.args.device)
        self.f1_metric_test.to(self.args.device)
        self.accuracy_metric.to(self.args.device)
        self.accuracy_metric_test.to(self.args.device)
        return 

    @typechecked
    def update_batch_level(self, loss, pred, target, level: Levels):
        '''
        update on the dataloader batch level 
        due to typechecked the levels from class Levels will throw error if wrong level is pases
        '''
        if level.value == 'train': 
            self.epoch_loss+= loss 
            if self.mode == 'cel': 
                self.accuracy_metric.update(pred, target)
                self.f1_metric.update(pred, target)
        else: 
            self.epoch_loss_test+= loss 
            if self.mode == 'cel': 
                self.accuracy_metric_test.update(pred, target)
                self.f1_metric_test.update(pred, target)

