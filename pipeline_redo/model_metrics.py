import logging 

class ModelMetrics: 
    '''
    class to deal with creating and tracking model training metrics
    - all of this functionality can be kept in train, but this makes the code more flexible and easy to understand 
    - tracks various metrics 
    - initialized in Train.train <- this class is unique to each call of train()
    
    metrics can change with output_mode classification or regression 
    '''

    def __init__(self, args): 
        self.args =args 
        # self.logger = logging.getLogger(self.args.logger_name)

    def initialize_metrics(self): 
        self.losses = []
        self.losses_test = []

        self.accuracy= []
        self.accuracy_test = []

        self.f1score = []
        self.f1score_test =[]

    

# mm= ModelMetrics(None)

# print(mm.__class__.__name__)