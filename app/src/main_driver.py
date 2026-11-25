'''
this contains the main driver for the entire application 
develop the structure, with api endpoints in mind
''' 

import logging 
from utils.schemas import ConfigSettingsBasemodel
from pipeline.utils import Utils
from utils.schemas import CustomerAttributes, ModelTrainingBasemodel, KMeansExplorationBasemodel
from utils.constants import ConstantFilenames

from pipeline.data_processing import DataProcessing
from pipeline.model_helper import ModelHelper
from pipeline.clustering import Clustering

from pipeline.predict import PredictCustomer

# global, can move to seperate file later on if there are more global states to track 
    # this is simple for now, so define here 
config_settings= ConfigSettingsBasemodel(logger_level='debug') 


class MainDriver: 
    '''
    main driver to interact from api 
    
    need function to be used before lifecycle yield 
    
    need the endpoints: 
        - calculate customer risk 
            given customer id, give the risk of opening credit for that customer
        - process data 
            process the raw data using the data pipeline 
            produce the k means curve plot to help choose k value when training  
        - explore kvalues 
            explore k values for kmeans clustering algo 
        - training models (from fresh or existing trained model)
            train a given model with specifications 
        - evaluate models 
            do evaluation scoring from the evaluation dataset on all 4 of the models 
    '''

    def __init__(self) -> None:
        self.logger =logging.getLogger('main_logger')
                
        pass

    # DONE 
    def start_app(self): 
        '''
        this should be used before yield call in api lifespan 
        start up app as needed 
        call only once 
        '''

        # make logger 
        HelperDriver.make_logger()

        # unzip data folder data.zip if needed 
        Utils.check_loaded_dataset()

        # make all stagnant paths as needed  
            # if filenames/paths don't exist, make them
            # also done for models 
        files = [file.value for file in ConstantFilenames]
        Utils.make_path(files )
        


    
    def calculate_customer_risk(self, customer: CustomerAttributes) -> tuple[bool, str]: 
        '''
        given customer id, give the risk of opening credit for that customer 
        '''
        # DONE 
        driver = PredictCustomer(customer)
        # pass onto rest of pipeline 

        if not driver.check_requirements(): 
            # raise HTTPException(status_code=404, detail="not all models are made")
            return False, "not all models are made"
        
        if not driver.check_customer(): # check if customer exists in df 
            # raise HTTPException(status_code=404, detail="customer id not found in dataset")
            return False, "customer id not found in dataset"
    
        pred = driver.get_prediction()
            # sets off pipeline 
        self.logger.info(f'customer {customer.id_number} is labeled as {pred}')
        return True, pred 
        pass 


    # DONE 
    def process_data(self) -> None: 
        '''
        process the raw data using the data pipeline 
        produce the k means curve plot to help choose k value when training
        
        this does: 
        data processing 
        kmeans k choosing  
        '''
        # this starts off the pipeline 
        self.logger.debug('begin data processing')
        dataprocessingclass = DataProcessing()
        # first, do data processing 
            # process the application record (x)
            # process the credit record (y)
            # get the joined data (x and y together)
            # save the joined data (split as needed further down the pipeline )
        dataprocessingclass.complete_data_processing()
        self.logger.debug('finished data processing')
        pass 

    # DONE
    def explore_kvalues(self, args: KMeansExplorationBasemodel): 
        '''
        explore k values for kmeans clustering algo 
        '''
        clustering = Clustering(args)
        clustering.driver()


    # DONE 
    def train_model(self, train_args: ModelTrainingBasemodel ) -> None: # TODO return type none?? 
        '''
        train a given model with specifications 
        specifications can be done with fastapi endpoint intake 
            which is taken in as UserSpecifiedTrainingBasemodel then converted to ModelTrainingBasemodel

        '''
        # then, do model building 
            # this is where the branching off has to be done 
            # build modular code so that i can reuse sections as needed 
        modelhelper = ModelHelper(train_args)
        modelhelper.driver()


        pass 




class HelperDriver: 
    @staticmethod
    def make_logger(): 
        '''
        initialize logger
        allow consol for logger 
    
        this should be done in main file, otherwise duplicate streamhandlers do duplicate printing 
        '''
        logger= logging.getLogger('main_logger') # keep logger name instead of __name__, this won't be a problem when imported - where __main__ becomes main
        # level_attr = getattr(logging, 'debug'.upper()) # similar tp logging.DEBUG
        level_attr = getattr(logging, config_settings.logger_level.upper())
        logging.basicConfig(filename ='logger.log', filemode='w',  level = level_attr)
            
        # needed for printing 
        console = logging.StreamHandler()
        console.setLevel(level_attr) # logging.DEBUG
        logger.addHandler(console)
    
        logger.debug('finshed logger initialization')


# from utils.constants import ModelType
# hd = MainDriver()
# hd.start_app()
# # hd.process_data()
# '''
# ### WORKING 
# # hd.train_model(ModelTrainingBasemodel(model_type= ModelType.fnn, lr = .0001, batch_size = 64, epochs= 15, load_model = False))
# # hd.train_model(ModelTrainingBasemodel(model_type= ModelType.fnn, lr = .001, epochs= 2, load_model = True))
# '''
# hd.train_model(ModelTrainingBasemodel(model_type= ModelType.slp, lr = .0001, batch_size = 64, epochs= 15, load_model = True))

# # hd.calculate_customer_risk(CustomerAttributes())

# # python -m src.main_driver