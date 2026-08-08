# use static_method for utils class 
import pandas as pd 
from sklearn.pipeline import Pipeline 
import torch 

import cloudpickle 

from pathlib import Path

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


    ### paths 
    @staticmethod 
    def make_path(location_list: list) -> None: 
        for location in location_list: 
            path = Path(location)
            path.parent.mkdir(parents= True, exist_ok=True )
        return 


    