


'''
data.zip 
  application_record.csv (constant)
  credit_record.csv (constant)

  data processing: ( later on: join application record and credit record, left join on id )
    split data as train, test 
    split as trainx,trainy, testx,testy 

    do pipeline on train x 
      application_record_pipeline.pkl (saved)

    transform: 
      trainx_processed.csv (saved)
      testx_processed.csv (saved)
    process: 
      trainy_processed.csv (saved)
      testy_processed.csv (saved)


note: when data drift is to be done, it will be done on the target variable 
the credit_record.csv will be updated with new values which can increase the risk score of an ID (month doesnt matter here)
'''


import logging 
from pathlib import Path 
import zipfile
import cloudpickle

from utils.constants import ConstantFilenames

import pandas as pd 
import numpy as np 
import torch

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import OrdinalEncoder , KBinsDiscretizer, OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer, QuantileTransformer
from sklearn.base import BaseEstimator


class DataProcessing: 
    '''
    this class sets up dataset folder from the application and credit records 

    data.zip has application_record.csv, credit_record.csv 

    load in data and make x, y data 
        will be trainx, testx, trainy, testx data 
    
    create pipeline for trainx 
    save the processed data 
    '''
    def __init__(self) -> None:
        self.logger = logging.getLogger('main_logger')

    def driver(self) -> None : 
        '''
        unzip data 
        load and split data
        do train test split 
            do pipeline on trainx 
            process trainx, testx, trainy, testy 
        '''

        self.unzip_data()

        # do work 
        self.processing()

        self.zip_data()

        return 



    def unzip_data(self) -> None : 
        ''' 
        unzips data.zip if needed 
        working dir = app 
        '''
        original_data = 'app/dataset/application_record.csv'
        path = Path(original_data)

        if path.is_file(): 
            self.logger.debug ('data.zip is unzipped already')

        else: 
            # unzip the data 
            zip_path = 'data.zip'
            extracted_to_path= './dataset/' # current directory 

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extracted_to_path)
            self. logger.info('data.zip is unzipped')

        return 

    def zip_data(self) -> None : 
        '''
        save the current dataset folder into data.zip
        '''

        folder= Path('dataset/')
        output_zip_folder = 'data.zip'

        with zipfile.ZipFile(output_zip_folder, 'w', compression= zipfile.ZIP_DEFLATED) as zip_ref: 
            for file in folder.rglob('*'): 
                if file.is_file(): # ignores subdirectories 
                    zip_ref.write(file, arcname=file.relative_to(folder))
                    # relative to keep internal folder structure 
        return 

    def processing(self): 
        '''
        load in original data
        process target 
        split data to xtrain, xtest, ytrain, ytest
        process inputs 
            fit transform on xtrain 
            transform on xtest 
        '''
        application_records= pd.read_csv(ConstantFilenames.application_record)
        credit_records = pd.read_csv(ConstantFilenames.credit_record)

        # do processing on credit_record to get credit risk
            # this should happen before joinin data due to need to sum over months of data per id 
        credit_records , y_pipeline= self.target_pipeline(credit_records)
        
        # join the data to shuffle properly 
        joined_data = pd.merge(application_records, credit_records, on= 'ID', how ='inner')

        x = joined_data.drop(labels = ['ID', 'STATUS'], axis = 1, inplace= False ) # make new df 
        y = joined_data[['STATUS']]


        # split the data 
        xtrain, xtest, ytrain, ytest= train_test_split(x, y, test_size=.3, 
                        random_state= 1, shuffle = True, 
                        stratify= y) # classification, keep frequencies the same 

        # transformations on x 
            # transformations on y already done 
        xtrain, x_pipeline= self.input_pipeline(xtrain) # this does fit transform 
        xtest= x_pipeline.transform(xtest)

        # now the data transformations are done 
            # train test split shuffle done for each call of DataProcessing.driver
            # save the following: y_pipeline, x_pipeline, xtrain, xtest, ytrain, ytest
        self.saving_data(y_pipeline, x_pipeline, xtrain, xtest, ytrain, ytest)


    ### processing functions 
    def saving_data(self, y_pipeline: BaseEstimator, x_pipeline: BaseEstimator, 
                    xtrain: pd.DataFrame, xtest: pd.DataFrame, 
                    ytrain: pd.DataFrame, ytest: pd.DataFrame) -> None : 
        '''
        this saves multiple things: 
            pipelines to pkl files 
            each df to its own csv file 
            makes tensors of each df and saves into 1 pytorch pt file (faster loading)
        '''

        # save pipelines 
            # note: need to use cloudpickle to save since there are custom methods/classes and also lambda is used 
            # cloudpickle is more thourough in saving than pickle 
        with open('dataset/processed/y_pipeline.pkl', 'wb') as file: 
            cloudpickle.dump(y_pipeline, file ) 
        with open('dataset/processed/x_pipeline.pkl', 'wb') as file: 
            cloudpickle.dump(x_pipeline, file )
        self.logger.debug('saved pipelines')


        # save data as csv 
        xtrain.to_csv('dataset/processed/xtrain.csv', index= False)
        xtest.to_csv('dataset/processed/xtest.csv', index= False)
        ytrain.to_csv('dataset/processed/ytrain.csv', index= False)
        ytest.to_csv('dataset/processed/ytest.csv', index= False)


        # save data as tensors (float 32 for inputs, long for targets)
        torch.save({
            'xtrain': torch.tensor(xtrain.values, dtype = torch.float32), 
            'xtest': torch.tensor(xtest.values, dtype = torch.float32), 
            'ytrain': torch.tensor(ytrain.values, dtype = torch.long), 
            'ytest': torch.tensor(ytest.values, dtype = torch.long)
        } , 'dataset/processed/torch_data.pt')
        self.logger.debug('saved data')

        return 
        
    def target_pipeline(self, records: pd.DataFrame)-> tuple[pd.DataFrame, BaseEstimator]: # can also return pipeline 
        '''
        this is the piepeline to transform y 
        
        it encodes the risk level to int 
        it groups over ID and summs the risk levels per month 
        then, the risks are binned to make this a classification problem 
        
        note: this has to be done before traintest split due, more in check_data.ipynb
        '''
        # encode the status 
        encode= ColumnTransformer(transformers=[
            ('encoded', OrdinalEncoder(categories= [['X', 'C', '0', '1', '2', '3', '4', '5']]) ,[ 'STATUS'])
            ], 
            remainder='passthrough', 
            verbose_feature_names_out=False).set_output(transform='pandas')

        # sum over the status to calcualte the risk 
        def get_risk(data: pd.DataFrame ) -> pd.DataFrame: 
            # always get copy 
            data  = data.copy(deep = True)

            # group by and get the summs over encoded status 
            scored_data = data.groupby('ID',as_index= False )['STATUS'].sum() # NOTE that this drops the months balance feature as wanted 
            return scored_data

        function_transformer_get_risk= FunctionTransformer(get_risk) # not ()

        discretize = ColumnTransformer(
            transformers=[
                ('keep id', 'passthrough', ['ID']),
                ('risk', KBinsDiscretizer(n_bins = 3, encode = 'ordinal', strategy= 'uniform'), ['STATUS'])
            ], 
            remainder= 'drop', 
            verbose_feature_names_out=False
        ).set_output(transform= 'pandas')


        pipeline = Pipeline(steps=[
            ('encode', encode), 
            ('sum_risk', function_transformer_get_risk), 
            ('bin', discretize)
        ]) 

        final_records= pipeline.fit_transform(records)
        self.logger.info('credit risk data processed ')

        return final_records, pipeline  # ID and STATUS columns, status = risk_score 



    def input_pipeline(self, apps: pd.DataFrame) -> tuple[pd.DataFrame, BaseEstimator]: 
        '''
        - encoding step first 
        ['CODE_GENDER' - binary 
        'FLAG_OWN_CAR', - binary 
        'FLAG_OWN_REALTY', - binary 
        'NAME_INCOME_TYPE', - onehot 
        'NAME_EDUCATION_TYPE', - ordinal 
        'NAME_FAMILY_STATUS', - onehot 
        'NAME_HOUSING_TYPE', - onehot 
        'OCCUPATION_TYPE'] - onehot, nans get encoded into their own column  


        - scaling step 
        ['ID', 
        'CNT_CHILDREN', - natural log + robust 
        'AMT_INCOME_TOTAL', - power transformer + robust 
        'DAYS_BIRTH', - minmax 
        'DAYS_EMPLOYED', - transformed into EMPLOYED (power transformer + robust), unemployed is -1 
        'FLAG_MOBIL', - dropped 
        'FLAG_WORK_PHONE', 
        'FLAG_PHONE',
        'FLAG_EMAIL',
        'CNT_FAM_MEMBERS'] - natural log + robust 
        - no need to scale binary encoded or already binary like the numerical flags 

        # all changes 

        - dropped FLAG_MOBIL 
        - OCCUPATION_TYPE 
        - 30 percent of the occupation type feature is missing, if the feature is generally uninformative in corr matrix, it can be removed -- no informative, keep in df 
        - NAME_EDUCATION_TYPE is ordinal encoding  
        - education_order = [
            'Lower secondary',               # 0: middle school /junior high
            'Secondary / secondary special', # 1: high school /vocational school
            'Incomplete higher',             # 2: some college/ university (no degree)
            'Higher education',              # 3: Bachelor's/ Master's degree
            'Academic degree'                # 4: PhD / doctorate /post-graduate degree
            ]


        - CODE_GENDER,FLAG_OWN_CAR, FLAG_OWN_REALTY is binary encoding (ordinal )
        - other categrorical features will be one hot encoded 

        numerical features: 
        - CNT_CHILDREN, CNT_FAM_MEMBERS = natural log + robust scaling for outiers 
        - AMT_INCOME_TOTAL = power transformer + robust 
        - DAYS_BIRTH = minmax for 0-1 scaling 
        - DAYS_EMPLOYED should be split into distribution for employed and another distribution for unemployed (positive value)
        - EMPLOYED power transformer + robust scaling for employed 
        - binary encoding for unemployed, this is done through setting 1 to unemployed 
        '''
        # preprocessing step 
            # drop FLAG_MOBIL
            # split DAYS_EMPLOYED

        def preprocess(data: pd.DataFrame) -> pd.DataFrame: 
            # always take copy, so that the df isnt changed in place 
            # func only for transform not fit 
            data = data.copy(deep = True)

            data['EMPLOYED'] = np.where( data['DAYS_EMPLOYED'] <=0, -data['DAYS_EMPLOYED'], -1 )
            data['UNEMPLOYED'] = np.where( data['DAYS_EMPLOYED'] > 0, 1, 0 ) # indicator binary variable 
                # turn into positive days employed and -1 = unemployed 

            data.drop(labels = ['FLAG_MOBIL', 'DAYS_EMPLOYED' ], axis  = 1 , inplace= True )
            return data 

        function_transfomer_preprocess= FunctionTransformer(preprocess)
        function_transformer_natural_log =FunctionTransformer( func = np.log1p) # not () for log call 

        # encoding step 
            # binary 
            # onehot 
            # ordinal 
            # prevent sparse 
        education_order = [
            'Lower secondary',               # 0: middle school /junior high
            'Secondary / secondary special', # 1: high school /vocational school
            'Incomplete higher',             # 2: some college/ university (no degree)
            'Higher education',              # 3: Bachelor's/ Master's degree
            'Academic degree'                # 4: PhD / doctorate /post-graduate degree
        ]

        
        squash_skew= ColumnTransformer( # will be put through robust scaler later down the pipeline 
            transformers= [
                ('squash_skewness', PowerTransformer(method = 'yeo-johnson', standardize=False), ['AMT_INCOME_TOTAL', 'EMPLOYED'] ), 
                            # this will sqash skewness for the numerical non min-maxed features 
                            # remaining tails are dealt with robust scaler 
                ('compress_outliers', function_transformer_natural_log,  ['CNT_CHILDREN', 'CNT_FAM_MEMBERS'])
            ], 
            remainder = 'passthrough', 
            verbose_feature_names_out=False 
        ).set_output(transform='pandas')


        encoders = ColumnTransformer(
            transformers=[
                ('binary', OrdinalEncoder(), ['CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY' ] ), # no missing vals in df no need to worry about encoding missing values 
                ('ordinal', OrdinalEncoder(categories = [education_order]), ['NAME_EDUCATION_TYPE']), 
                ('one_hot', OneHotEncoder(sparse_output=False), ['NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE'])
            ], 
            remainder= 'passthrough', 
            verbose_feature_names_out=False # auto for onehot encoded 
        ).set_output(transform='pandas')

        scalers = ColumnTransformer(
            transformers = [
                ('robust', RobustScaler(), ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS', 'EMPLOYED']),  # try this
                ('minmax', MinMaxScaler(), ['DAYS_BIRTH']),
                ('standard_scale_ordinal', StandardScaler(), ['NAME_EDUCATION_TYPE']) 
            ], 
            remainder = 'passthrough', 
            verbose_feature_names_out= False 
        ).set_output(transform='pandas')


        pipeline = Pipeline(
            steps = [
                ('column_preprocessing', function_transfomer_preprocess), 
                ('squash_skeq', squash_skew), # ignore for now 
                ('encoders', encoders), 
                ('scalers', scalers), 
                ('standard_scaling', QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=1))
            ]
        ).set_output(transform = 'pandas')

        transformed_df = pipeline.fit_transform(apps)
        self.logger.info('credit card applications data processed ')

        return transformed_df, pipeline 

    
# dp= DataProcessing()

# # dp.zip_data()
# dp.unzip_data()


# python -m src.data_scripts.data_processing

