
from pydantic import BaseModel, model_validator 
from typing import Literal # Self, for 3.11 or higher 

from utils.constants import ModelType

# TODO make classes for settings, data processing, and model training- used for configuration

class ConfigSettings(BaseModel): 
    '''
    no need to read in .env file here, there is no api key or personal info
    it is enough just to specify the basemodel and read in the data 
    # TODO create .env pipeline if api key gets made 
    '''
    logger_level: Literal['info', 'debug']

config_settings= ConfigSettings(logger_level='debug')


class DataProcessingBasemodel(BaseModel): 
    '''
    this is a basemodel rather than enum class 
    this allows for high flexibility in creating various models from differently processed data 

    class to deal with data processing pipeline configuraiton 
    '''
    # customer credit application file - raw data 
    application_record: str = 'dataset/application_record.csv'
    # customer history - raw data 
    credit_record: str = 'dataset/credit_record.csv' 

    # processed data 
    # saved pipeline
    data_pipeline:str = 'model_info/data_processing/application_record_pipeline.pkl'

    # saved processed data file 
    data_filename:str = 'model_info/data_processing/data.csv'
    
'''
for the training specifications to be done, there are some fields which are dependent on others 
this means that the best thing to do is inherit basemodel classes 
the api will take in the input basemodel 
    then, it will apply it to the larger model training basemodel to apply transfomations
NOTE this seems to be the best way to do this, can adjust later on as needed 
'''

class UserSpecifiedTrainingBasemodel(BaseModel): # subclass 
    '''
    this model is to be used in the api endpoint intake 
    then, the api function will transform it to the larget model training basemodel 
        this applies the needed transformations 
        without confusing the user with too many options 

    # CALL note that all fields have defaults 
    '''

    # refer with either name or call 
    model_type: ModelType = ModelType.slp # CALL with .name or .value as needed 

    # model filename and plotname the the model will be saved in 
    model_name: str | None  = None # optional to pass in, but can be made into default later on 
    

    ### training loop specifications ------ 
    load_model: bool = False  # decide to continue training from loaded model 

    batch_size: int = 1000 
    lr: float = .0001 
    epochs: int = 3 
    # not needed for all models
    hidden_sizes :list[int] = [30, 20 ]
    k_value: int= 11 


    '''
    trying to use model validator 
    can have multiples of them, either in before or after mode
    can also use field validator 
        this uses classmethod because it is done before the class is created 
        model validator is done after the class is created, so it can use self -> self / str(class name)
    '''
    @model_validator(mode = 'after')
    def add_in_defaults(self) ->'UserSpecifiedTrainingBasemodel':  # Self: or -> 'ModelTrainBasemodel' : 
        if not self.model_name: # recieves optional none 
            # then get the default filename associated with the model type 
            self.model_name = self.model_type.value # string 
        return self 
            

    
class ModelTrainingBasemodel(UserSpecifiedTrainingBasemodel): # superclass 
    '''
    this is used in modelhelper to train the various models 
    all needed args are available through this class instance 
    '''
    # these fields will be automatically specified below, no need to write out, they will be overridden anyways
    model_filename : str|None  = None 
    plot_filepath: str|None = None 

    # output mode = classification by default 

    # stagnant, user doesn't specify
    # for clustering model saving: 
    elbow_curve_for_kmeans: str = 'images/kmeans_plots/elbow_curve_for_kmeans.png' # plot 
    kmeans_model_filename: str = 'model_info/saved_models/clustering/kmeans_model.pkl' # model 
    autoencoder_kmeans_model_filename: str = 'model_info/saved_models/clustering/autoencoder_kmeans_model.pkl' # model 

    
     
    @model_validator(mode = 'after')
    def add_in_defaults(self) ->'ModelTrainingBasemodel':  # Self: or -> 'ModelTrainBasemodel' : 
        '''
        use model validator to auto populate the model filename and plotpath
        base the file name and plot nambe based on the model_name 
        '''
        self.model_filename = f'model_info/saved_models/model_{self.model_name}.pt'
        self.plot_filepath = f'images/nn_plots/{self.model_name}' # .csv added in code # TODO 

        return self 

    








