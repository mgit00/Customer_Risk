import pandas as pd 
from .utils import Utils 

from utils.constants import ConstantFilenames
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import FunctionTransformer, FeatureUnion

from sklearn.base import BaseEstimator , TransformerMixin

import logging 
'''
process the customer records dataset
'''

class DataProcessing: 
    '''
    application_record -> applications_processed 
    credit_record -> customer_risk 
    '''
    def __init__(self): 
        self.logger= logging.getLogger('main_logger')
        self.output_mode= 'classification'

    def complete_data_processing(self): 
        '''
        calls the needed functions to fully process data 
            process the application record (x)
            process the credit record (y)
            get the joined data (x and y together)
            save the joined data (split as needed further down the pipeline )
        '''

        self.logger.debug('doing full data processing ')

        og_credit_record= Utils.read_data(filename=ConstantFilenames.credit_record.value)
        og_application_record = Utils.read_data(filename=ConstantFilenames.application_record.value)

        customer_risk = self.customer_risk_processing(og_credit_record)
        applications, applications_pipeline= self.cc_applications_processing(og_application_record)

        # customer risk processing has no pipeline, but applications does, save pipeline 
        Utils.save_pipeline(pipeline= applications_pipeline, location = ConstantFilenames.data_pipeline.value)
        self.logger.info('credit application pipeline saved: ' + ConstantFilenames.data_pipeline.value)

        # make the joined data and save it 
        joined= self.join_data(customer_risk, applications)
            # remove id column 
        joined.drop(columns = ['ID'], axis = 1 , inplace= True )
        Utils.save_df(df= joined, filename =ConstantFilenames.data_filename.value)
        self.logger.info('x, y data saved to: '+ ConstantFilenames.data_filename.value)


    ### processing pipelines 
    def customer_risk_processing(self, records ): 
        '''
        the risk score is calculated 
        - encode status into 0-5 score 
        - sum over all months for each customer 
        - discretize into 5 categories <- categorical encoding 
        '''
        # transform STATUS into ordinal levels  
        categories= ['X' , 'C' ] +  [str(i) for i in range(0, 6 )]
        encoder= OrdinalEncoder(categories= [categories])
        records['status_encoded'] = encoder.fit_transform(records[['STATUS']])

        # group by unique customer ID 
        records_grouped= records.groupby(by= 'ID' , axis = 0)
        # sum over each group with agg() 
            # use transform() if mapping back to ungrouped records df 
        creditworthiness= records_grouped.status_encoded.agg(summed= 'sum')
        # make ID feature from index 
        creditworthiness.reset_index(drop= False, inplace= True )

        if self.output_mode == 'classification': 
            # discreditize into bins 
            score_mean= creditworthiness.summed.where(creditworthiness.summed > 0 ).mean()
            score_std= creditworthiness.summed.where(creditworthiness.summed > 0 ).std()
                # pd.cut is right selctive ( , ] ranges with right= True setting 
            ranges= (-float('inf'),0, score_mean- score_std, score_mean, score_mean + score_std, float('inf'))
            range_titles= range(0, 5) # range_titles= ('none', 'low', 'medium_low', 'medium_high', 'high')
            creditworthiness['risk_level']= pd.cut(creditworthiness['summed'], bins=ranges, labels=range_titles, right= True)
            
            # drop unneded cols 
            creditworthiness.drop(labels=['summed'], axis = 1, inplace=True )

        else : # regression
            creditworthiness.rename( {'summed': 'risk_level'}, axis = 1, inplace= True )

        self.logger.info('customer risk data processed' )
        return creditworthiness

    def cc_applications_processing(self, apps ) -> tuple: # pd.DataFrame, pipeline: 
        '''
        this will use sklearn Pipeline for ease of replication 
            # ensure all have fit and transform/predict functions associated with them 
        the Pipeline will also be saved for ease later on, allowing for more functionality 

        note: label_encoder()
        pipeline from sklearn doesn't work properly with LabelEncoder() because of the inputs taken from fit and transform. 
        pipeline requires fit(x, y = None), whereas LabelEncoder().fit(x)
        this is because labelencoder is meant to be used for y target feature, whereas ordinalencoder is supposed to be used with x input data
        this means, I have to create a wrapper for the label encoder for this to work or i can just use ordinal encoder 
        '''
        
        encoder = self.encoder_pipeline()
        scaler= self.scaler_pipeline()
        drops= self.drops_pipeline()

        # nested pipelines
        pipeline = Pipeline(steps=[ 
            ('encoding', encoder),  # column transformer 
            ('scaling', scaler), # column transformer 
            ('drop', drops) # class I designed myself with the fit, transform, get feature names -> column transformer 
        ])
        pipeline.set_output(transform='pandas')

        apps= pipeline.fit_transform(apps)
        
        self.logger.info('credit card applications data processed ')
        return apps, pipeline 
    

    def join_data(self, customer_risk, applications) -> pd.DataFrame: 
        ''' 
        merge the data based on id 
        - note that mergeing is for columns and join is for indecies 
        '''
        res = customer_risk.merge(applications, on='ID', how= 'inner') 
        # drop the ID column later on in the process 

        self.logger.debug('full joined x,y data done')
        return res 


    ### different sub-pipelines for cc_applications processing
    def encoder_pipeline(self): 
        '''
        this creates the pipeline for encoding the cols 
        '''

        # convert categroical features 
        onehot= ['OCCUPATION_TYPE' ]
        nominal = ['CODE_GENDER', 'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS',
                   'NAME_HOUSING_TYPE']
        ordinal= ['FLAG_OWN_CAR' ,'FLAG_OWN_REALTY', 'NAME_EDUCATION_TYPE']
        ordinal_values= [['N', 'Y'], 
                         ['N', 'Y'], 
                         ['Lower secondary','Secondary / secondary special',
                          'Incomplete higher',  'Higher education', 'Academic degree'] 
                          ]
        
        # make pipeline for nominal features 
        nominal_preprocessor= Pipeline(steps=[
            ('missing_vals', SimpleImputer(fill_value= 'Missing', strategy= 'constant')), # for missing vals 
            ('nominal_encoder', OrdinalEncoder()) # LabelEncoder()- feature transformations, read why this is done above ... 
        ])

        # make pipeline for ordinal features 
        ordinal_preprocessor = OrdinalEncoder(categories=ordinal_values)

        # make pipeline for onehotencoding features 
        onehot_preprocessor = OneHotEncoder(sparse_output= False) # set_output already handled below 

        # apply cols to right pipelines with columntransformer 
        column_transformer= ColumnTransformer(transformers=[
            ('ordinals', ordinal_preprocessor, ordinal), 
            ('nominals', nominal_preprocessor, nominal), 
            ('onehots', onehot_preprocessor, onehot)
        ], 
        remainder= 'passthrough', 
        verbose_feature_names_out= False) # to get og column names (for purpose of getting back pd df )

        self.logger.debug('encoder pipeline done ')
        return column_transformer



    def scaler_pipeline(self): 
        '''
        this creates the pipeline for scaling the features 
        '''

        robust_feats= ['CNT_CHILDREN', 'DAYS_EMPLOYED']
        standard_feats = ['AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'CNT_FAM_MEMBERS'  ]
        binary_feats = ['unemployed']  # to be created from DAYS_EMPLOYED 

        # transform DAYS_EMPLOYED into new binary feature with feature_union 
        identiy_transform= FunctionTransformer(func= None, feature_names_out='one-to-one')

        def employment_condition(df): 
            res= df['DAYS_EMPLOYED'].apply(lambda row: 1 if row > 0 else 0 ) # series is returned 
            return res .values # want array to be returned, no issues with name indexing if names are removed 
        def feat_name_fx(func_transformer_instance, cols_feature_names): 
            return binary_feats # need to return list of feature names (same as number of cols returned in func)
        new_transform = FunctionTransformer(func= employment_condition, feature_names_out=feat_name_fx )


        featureunion= FeatureUnion( transformer_list= [
            ('identity', identiy_transform), 
            ('new_feat_unemployed', new_transform)
        ], 
        verbose_feature_names_out = False ) # want simpler names 

        # define the scalers (or use function transformer for my own scaler)
        robust=RobustScaler()
        standard= StandardScaler()
        columntransformer = ColumnTransformer(transformers=[
            ('robust_scaling', robust, robust_feats), 
            ('standard_scaling', standard, standard_feats)
        ], 
        remainder='passthrough',
        verbose_feature_names_out= False )

        pipeline = Pipeline(steps=[
            ('binary_transform', featureunion), # df with new feature for unemployed 
            ('column_transforms', columntransformer), # robust and standard scaling with targeted cols 
        ])

        self.logger.debug('scaler pipeline done ')

        return pipeline

    def drops_pipeline(self): 
        '''
        this creates the pipeline for encoding the cols 
        - I want practice making my own pipeline step, so I will make dropping the mobile flag as its own step in the pipeline


        note that it is also possible to do this, but I wanted practice with making custom transformers 
        # columns to drop
        columns_to_drop = ['feature', 'feature_next']

        # pipeline with ColumnTransformer to drop cols
        processor = ColumnTransformer(transformers=[
                ('column_drop', 'drop', columns_to_drop),
            ], remainder= 'passthrough')
        '''

        cols_dropping= ['FLAG_MOBIL']

        class CustomDrops(BaseEstimator, TransformerMixin): # self-created custom class 
            def __init__(self, drop_cols): # no args... 
                self.drop_cols= drop_cols 
                self.feat_names= [] # feature_names_in 

            def fit(self, X, y= None ): 
                # learn the feature names 
                self.feat_names= X.columns
                return self # nothing to be learned  
            def transform(self, X: pd.DataFrame, y= None ): 
                X_new= X.drop(columns= self.drop_cols, axis= 1,  inplace= False ) # make a copy to not disturb the original 
                self.feat_names= X_new.columns
                return X_new
            
            def get_feature_names_out(self, input_feats_unassigned): # needed for set_output() functoin for pipeline 
                return self.feat_names 
            

        columntransformer= ColumnTransformer(transformers=[
            ('dropping_cols', CustomDrops(cols_dropping), cols_dropping)
        ], 
        remainder='passthrough', 
        verbose_feature_names_out=False)

        self.logger.debug('self made pipeline (drops) done ')
        return columntransformer 


    ### pipeline just used for testing  
    def encoder_pipeline_simple(self): 
        '''
        this creates the pipeline for encoding the cols 
        - no one hot encoding (for occupation_type), this is done in the similar function above 
        '''

        # convert categroical features 
        nominal = ['CODE_GENDER', 'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS',
                   'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']
        ordinal= ['FLAG_OWN_CAR' ,'FLAG_OWN_REALTY', 'NAME_EDUCATION_TYPE']
        ordinal_values= [['N', 'Y'], 
                         ['N', 'Y'], 
                         ['Lower secondary','Secondary / secondary special',
                          'Incomplete higher',  'Higher education', 'Academic degree'] 
                          ]
        
        # make pipeline for nominal features 
        nominal_preprocessor= Pipeline(steps=[
            ('missing_vals', SimpleImputer(fill_value= 'Missing', strategy= 'constant')), # for missing vals 
            ('nominal_encoder', OrdinalEncoder()) # LabelEncoder()- feature transformations, read why this is done above ... 
        ])

        # make pipeline for ordinal features 
        ordinal_preprocessor = OrdinalEncoder(categories=ordinal_values)


        # apply cols to right pipelines with columntransformer 
        column_transformer= ColumnTransformer(transformers=[
            ('ordinals', ordinal_preprocessor, ordinal), 
            ('nominals', nominal_preprocessor, nominal), 
        ], 
        remainder= 'passthrough', 
        verbose_feature_names_out= False) # to get og column names (for purpose of getting back pd df )

       
        return column_transformer


    

    

        

