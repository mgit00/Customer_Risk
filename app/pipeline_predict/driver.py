import logging 
from pathlib import Path 

from .customer_attributes import CustomerAttributes
from .predict import Predict, PredictEvaluation

from pipeline.utils import Utils 
import pandas as pd

class DriverPrediction: 
    '''
    this will drive the prediction steps for the customer given
    the customer will be given a score based on each of the 4 classification models
    '''

    def __init__(self, customer: CustomerAttributes= CustomerAttributes()): 
        # the logger is already given at name main_logger 
        self.logger= logging.getLogger('main_logger')

        self.customer = customer 
        self.data = Utils.read_data(filename='./dataset/application_record.csv')

        self.predict = Predict(self.logger)
    
    def check_requirements(self) -> bool : 
        '''
        ensure that the following are present in the directory: 
            - application record pipeline
            - slp classification model 
            - fnn classification model 
            - kmeans+ fnn classification model 
            - autoencoder+ kmeans+ fnn classification model
            - both kmeans models 
        '''

        paths= ['./model_info/data_processing/application_record_pipeline.pkl', 
                './model_info/saved_models/model_classification_slp.pt', 
                './model_info/saved_models/model_classification_fnn.pt', 
                './model_info/saved_models/model_classification_kmeans+fnn.pt', 
                './model_info/saved_models/model_classification_autoencoder+kmeans+fnn.pt', 
                './model_info/saved_models/clustering/kmeans_model.pkl', 
                './model_info/saved_models/clustering/autoencoder_kmeans_model.pkl']
        for path in paths: 
            if Path (path).is_file(): 
                pass 
            else : 
                self.logger.info('issue- path is not found for: ' + path)
                return False 
            
        self.logger.debug('all required paths are found')
        return True 
    
    
    def check_customer(self) ->  bool : 
        return self.customer.id_number in self.data.ID.values
      
     
    def get_prediction(self) -> str : 
        '''
        does the data processing and prediction step 
        '''
        data = self.data[self.data.ID == self.customer.id_number].iloc[[0],:] # get the first instance

        data= self.predict.process_data(data)

        prediction = self.predict.predict_step(data)
        return prediction

    def evaluate_model(self) -> str : 
        '''
        make dataset to test the model accuracy 

        does the data processing and prediction step 
        '''
        predict = PredictEvaluation(self.logger)
        testing_data = pd.read_csv('dataset/data.csv')

        data = self.data[self.data.ID.isin(testing_data['ID'])]

        data= predict.process_data(data)

        prediction = predict.predict_step_evalset(data)

        res= predict.score_predictions(testing_data['risk_level'], prediction)
        return res

