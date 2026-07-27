# use static_method for utils class 
import pandas as pd 
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer

class Utils: 
    '''
    helper functions which can be reused as needed 
    '''

    @staticmethod
    def save_df(df:pd.DataFrame, filename): 
        df.to_csv(filename, index = False  )

    @staticmethod
    def read_data(filename) -> pd.DataFrame: 
        return pd.read_csv(filename)

    @staticmethod
    def pipeline_creator(steps_list:list): 
        pipeline= Pipeline(steps= steps_list )
        pipeline.set_output(transform='pandas')

        return pipeline 

    