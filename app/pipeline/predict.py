from utils.constants import ConstantFilenames
from utils.schemas import CustomerAttributes

from pipeline.models import AutoencoderKmeansFNN
from pipeline.utils import Utils

import logging
from pathlib import Path 

import pandas as pd 
import torch 
import pickle 


class PredictCustomer: 
    ''' this class predicts the risk level of the customer '''
    def __init__(self, customer: CustomerAttributes ) -> None:
        '''
        use autoencoder kmeans 
            check that data pipeline and model is made 
            check customer is in dataset 
                or give data 
            process the data 
            load in the autoencer kmeans model 
            initalize the model 
            give the loaded state dict to model, and set to eval mode 
            do prediction 
            get argmax for class to get risk labeling 
        '''
        self.logger= logging.getLogger('main_logger') 
        self.level_options= {0: 'no risk', 
                         1: 'low risk', 
                         2: 'high risk'} # ('none', 'low', 'high')
        self.input_size, self.output_size = 35, 3 

        self.customer = customer 

        self.model_path = 'data_saved/trained_models/autoencoder_kmeans_fnn/model.pt'
        self.kmeans_model_path = 'data_saved/trained_models/autoencoder_kmeans_fnn/kmeans_model.pt'


        self.data_path = ConstantFilenames.application_record.value # raw data 
        self.data_pipeline = ConstantFilenames.data_pipeline.value # created data processigg pipeline 
        self.data = pd.read_csv(self.data_path)

    def check_requirements(self) -> bool: 
        for path in [self.model_path, self.data_path, self.data_pipeline]: 
            if Path (path).is_file(): 
                pass 
            else : 
                self.logger.info('issue- path is not found for: ' + path)
                return False 
            
        self.logger.debug('all required paths are found')
        return True 

    def check_customer(self) -> bool: 
        return self.customer.id_number in self.data.ID.values

    def get_prediction(self) -> str : 
        '''
        does the data processing and prediction step 
        '''
        customer_data = self.data[self.data.ID == self.customer.id_number].iloc[[0],:] # get the first instance

        # put the data through the created data pipeline 
        pipeline = Utils.load_pipeline(self.data_pipeline)

        data = pipeline.transform(customer_data)
        data.drop(columns = ['ID'], axis = 1 , inplace= True )
        # data is now ready for passing into the models 
        self.logger.info('data is now put through the previously created data processing pipeline')
        
        x = torch.tensor(data.values, dtype = torch.float32) # removed values bc of scaling 
        self.logger.debug('data now transformed to tensor')

        
        # load in the models 
        loaded = Utils.load_model(location = self.model_path)
        hidden_sizes= loaded['args'].hidden_sizes 
        k_value= loaded['args'].k_value
        # autoencoder_kmeans_model_filename= loaded['args'].autoencoder_kmeans_model_filename

        model = AutoencoderKmeansFNN(k_value, self.input_size, hidden_sizes, self.output_size) 
        # note no need for optimizer, only for training to calc gradients and update weights
        
        # return self.do_predict(data, model, loaded)
        model.load_state_dict(loaded['model_state_dict'])
        model.kmeans = Utils.load_kmeans_model(self.kmeans_model_path)
        self.logger.debug('autoencoder kmeans fnn model loaded in ')

        model.eval()
        self.logger.debug('prediction done')

        pred =model(x) # forward pass 
        
        level = torch.argmax(pred).item()
        return self.level_options[level]
    


