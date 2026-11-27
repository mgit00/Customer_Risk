
import logging 

# use static_method for utils class 
import pandas as pd 
from sklearn.pipeline import Pipeline 
import torch 

import cloudpickle 
import pickle 

from pathlib import Path
import zipfile


class Utils: 
    '''
    helper functions which can be reused as needed 
    '''
    ### data 
    @staticmethod
    def save_df(df:pd.DataFrame, filename): 
        df.to_csv(filename, index = False  )

    @staticmethod
    def read_data(filename) -> pd.DataFrame: 
        return pd.read_csv(filename)


    ### pipelines
    @staticmethod
    def pipeline_creator(steps_list:list): 
        pipeline= Pipeline(steps= steps_list )
        pipeline.set_output(transform='pandas')

        return pipeline 
    
    @staticmethod
    def save_pipeline(pipeline, location ): 
        '''
        need to use cloudpickle to save since there are custom methods/classes and also lambda is used
        cloudpickle is more thourough in saving than pickle 
        '''
        with open(location, 'wb') as filename: 
            cloudpickle.dump(pipeline, filename)

    @staticmethod 
    def load_pipeline(location): 
        with open(location, 'rb') as filename: 
            pipeline = cloudpickle.load(filename)

        return pipeline 


    ### models 
    @staticmethod
    def save_model(location, model_state): 
        torch.save(model_state, location) 
        print('model state saved to: ', location)

    @staticmethod
    def load_model(location): 
        model_state  = torch.load(location, weights_only = False )
        print('model loaded in from: ', location)
        return model_state
    
    @staticmethod
    def save_kmeans_model(location, model_state): 
        with open(location, "wb") as f:
            pickle.dump(model_state, f)
        print('kmeans model state saved to: ', location)

    @staticmethod
    def load_kmeans_model(location): 
        with open(location, "rb") as f:
            loaded_kmeans = pickle.load(f)
        print('kmeans model loaded in from: ', location)
        return loaded_kmeans
    


    ### paths 
    @staticmethod 
    def make_path(location_list: list) -> None: 
        for location in location_list: 
            path = Path(location)
            path.parent.mkdir(parents= True, exist_ok=True )
        return 
    @staticmethod 
    def make_folder(location_list: list) -> None: 
        '''path.parent not used'''
        for location in location_list: 
            path = Path(location)
            path.mkdir(parents= True, exist_ok=True )
        return 


    @staticmethod
    def check_loaded_dataset() -> None: 
        # FIXME do this for dataset.zip into dataset_/ folder 
        #   this will look at new processed data 
         
        logger= logging.getLogger('main_logger')

        data_file = "dataset/application_record.csv"
        file_path = Path(data_file)

        if file_path.is_file(): 
            logger.debug('dataset already unzipped')
            pass  
        else: 
            # add the file
            zip_path = "./data.zip"
            extract_path = "./"  # current directory
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)
            logger.info('dataset folder unzipped')