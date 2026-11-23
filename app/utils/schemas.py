
from pydantic import BaseModel, model_validator, PrivateAttr, computed_field
from typing import Literal # Self, for 3.11 or higher 

from utils.constants import ModelType

# DONE make classes for settings, data processing, and model training- used for configuration

class ConfigSettingsBasemodel(BaseModel): 
    '''
    no need to read in .env file here, there is no api key or personal info
    it is enough just to specify the basemodel and read in the data 
    # TODO create .env pipeline if api key gets made 
    '''
    logger_level: Literal['info', 'debug']


class CustomerAttributes(BaseModel): 
    '''
    create customer attribute model used for the fastapi connection

    gets the customer id 
    # instantiate with ** {dictionary}
    # access with CustomerAttributes_instance.key_value 
    '''
    id_number: int =  5008804


class KMeansExplorationBasemodel(BaseModel): 
    '''
    this model is used to explore choices of k for the kmeans model
    '''
    # get filename for where data is saved 
    k_values : tuple[int, int] = (2, 30)


'''
for the training specifications to be done, there are some fields which are dependent on others 
this means that the best thing to do is inherit basemodel classes 
the api will take in the input basemodel 
    then, it will apply it to the larger model training basemodel to apply transfomations
NOTE this seems to be the best way to do this, can adjust later on as needed 
'''
class ModelTrainingBasemodel(BaseModel): # subclass 
    '''
    this model is to be used in the api endpoint intake 
    then, the api function will transform it to the larget model training basemodel 
        this applies the needed transformations 
        without confusing the user with too many options 

    # CALL note that all fields have defaults 
    '''

    # refer with either name or call 
    model_type: ModelType = ModelType.slp # CALL with .name or .value as needed     

    ### training loop specifications ------ 
    load_model: bool = False  # decide to continue training from loaded model 

    batch_size: int = 1000 # 32 1000 before, causing majority class collapse 
    lr: float = .0001 
    epochs: int = 3 
    # not needed for all models
    hidden_sizes :list[int] = [30, 20 ]
    k_value: int| None = 11 


    @computed_field
    @property 
    def kmeans_model_filename(self) -> str: # optionally used in code depending on model
        ''' associated kmeans trained model'''
        return f'data_saved/trained_models/{self.model_type.name}/kmeans_model.pt' 
    
    @computed_field # made on the fly after class init, not stored in memory
    @property # python function to make calls to functions without ()
    def model_filename(self) -> str: # type 
        ''' nn trained model '''
        return f'data_saved/trained_models/{self.model_type.name}/model.pt'

    @computed_field
    @property 
    def plot_filepath(self) -> str: 
        ''' model training plots '''
        # the last filename is ignored, so add this for correct path 
        # this is done in plotting function, contains plot name and .csv extension 
        return f'data_saved/trained_models/{self.model_type.name}/plot_' 
    

    

        

    '''
    notes: 
    trying to use model validator for dynamic changes 
    can have multiples of them, either in before or after mode
    can also use field validator 
        this uses classmethod because it is done before the class is created 
        model validator is done after the class is created, so it can use self -> self / str(class name)
    can also try ClassVar or PrivateAttr or computed_field and property 
    '''
    # model_name: str | None  = None # optional to pass in, but can be made into default later on 
    # @model_validator(mode = 'after')
    # def add_in_defaults(self) ->'ModelTrainingBasemodel':  # Self: or -> 'ModelTrainBasemodel' : 
    #     if not self.model_name: # recieves optional none 
    #         # then get the default filename associated with the model type 
    #         self.model_name = self.model_type.value # string 
    #     return self 

    # model validator run after the class is spun up, and then kept in memory 
    # for unchanging vars to be stored in memory, use private/class attr - not available in modeldump
        # priavte is for specific instance, class is for when the class file is imported 
    # can also do model validator with None init, but this allows user to input their values 
    # other way is to use computed field and property to bring vars not read in 
        # through model dump (private/class attr) and bring it to the readable front 
    # filepaths will be based on modelname 

    # these fields will be automatically specified below, no need to write out, they will be overridden anyways
    # model_filename : str|None  = None 
    # plot_filepath: str|None = None 
    # @model_validator(mode = 'after')
    # def add_in_defaults(self) ->'ModelTrainingBasemodel':  # Self: or -> 'ModelTrainBasemodel' : 
    #     '''
    #     use model validator to auto populate the model filename and plotpath
    #     base the file name and plot nambe based on the model_name 
    #     '''
    #     self.model_filename = f'model_info/saved_models/model_{self.model_name}.pt'
    #     self.plot_filepath = f'images/nn_plots/{self.model_name}' # .csv added in code # TODO 

    #     return self 


    # privateattr is made after class init and stored in memory
        # computedfield + property is made on the fly and not stored in memory, accessible after init only 
    # private_model_filename: str = PrivateAttr(default = '') # note can remove, no need to set seperate private var definition 
    

    

            



    








