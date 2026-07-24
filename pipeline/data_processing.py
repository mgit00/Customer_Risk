import pandas as pd 
from utils import Utils 

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

'''
process the customer records dataset
'''

class DataProcessing: 
    '''
    application_record -> applications_processed 
    credit_record -> customer_risk 
    '''
    def __init__(self, args): 
        self.args= args 

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

        # discreditize into bins 
        score_mean= creditworthiness.summed.where(creditworthiness.summed > 0 ).mean()
        score_std= creditworthiness.summed.where(creditworthiness.summed > 0 ).std()
            # pd.cut is right selctive ( , ] ranges with right= True setting 
        ranges= (-float('inf'),0, score_mean- score_std, score_mean, score_mean + score_std, float('inf'))
        range_titles= range(0, 5) # range_titles= ('none', 'low', 'medium_low', 'medium_high', 'high')
        creditworthiness['risk_level']= pd.cut(creditworthiness['summed'], bins=ranges, labels=range_titles, right= True)
        
        # drop unneded cols 
        creditworthiness.drop(labels=['summed'], axis = 1, inplace=True )

        return creditworthiness



    def cc_applications_processing(self, apps ) -> pd.DataFrame: 
        '''
        this will use sklearn Pipeline for ease of replication 
            # ensure all have fit and transform/predict functions associated with them 
        the Pipeline will also be saved for ease later on, allowing for more functionality 

        note: label_encoder()
        pipeline from sklearn doesn't work properly with LabelEncoder() because of the inputs taken from fit and transform. 
        pipeline requires fit(x, y = None), whereas LabelEncoder().fit(x)
        this is because labelencoder is meant to be used for y target feature, whereas ordinalencoder is supposed to be used with x input data
        this means, I have to create a wrapper for the label encoder for this to work or i can just use ordinal encoder 


        OCCUPATION_TYPE <<-- shoudl be 1 hot encoded ....
        '''
        # keep column names in memory
        cols= apps.columns

        # convert categroical features 
        nominal = ['CODE_GENDER', 'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS',
                   'NAME_HOUSING_TYPE','OCCUPATION_TYPE' ]
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
            ('nominals', nominal_preprocessor, nominal)
        ], 
        remainder= 'passthrough', 
        verbose_feature_names_out= False) # to get og column names (for purpose of getting back pd df )

        # place columntransformer into overarching pipeline 
            # remove the flag_mobil feature, do onehot encoding for occupation_type 
        transformer_pipeline= Pipeline(steps=[
            ('data_transform',  column_transformer) # data transformation step 
        ])

        # fit and predict 
        transformer_pipeline.fit(apps )
        res = transformer_pipeline.transform(apps)

        # the cols are now rearranged because of the columntransformer
        # reassign and rearrange the cols 
        res = pd.DataFrame(res, columns= column_transformer.get_feature_names_out()) # array  
        res= res[cols]
         
        return res

    

        

