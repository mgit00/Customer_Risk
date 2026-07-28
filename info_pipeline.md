# pipeline for the credit risk model 

- first explore data 
- process data 
    - encode data 
    - scale (minmax, normalize)

## ultimate goal: 
- create a classifier which states the risk of not paying bach credit card level of the applicant
- over time, this can be integrated into the larger credit card approval system 

to create a credit risk classifier, the following has to be done: 
- create plain classifier 
    - this will train on joined application and record data for the customer. 
    - from this, the customers without a record can be clasified into risk category. 


- create kmeans clusters and combine with classifier 
- create autoencoder, kmeans clusters, then classifier 


# directory 
- dataset: contains all datasets 
    - application_record.csv = original application data 
    - credit_record.csv= original customer credit records data 
    - customer_risk.csv= unique customer credit risk/creditworthiness rating 

## pipeline 
- pipeline 
    - argparse 
    - calls each of the files needed 
        - calls and interacts with directory (file saving)
        - segments calls as needed, determined by argparse 
- data_processing 
    - processes the data with pipeline from sklearn 
    - uses: Pipeline, ColumnTransformer, FeatureUnion, FunctionTransformer, self-crated custom classes created for Pipeline using BaseEstimator/TransformerMixin
        - gained experience nesting pipelines
    - uses various encoding and scaling methods 
        - specific methods chosen based on work in data_exploration.ipynb 
- utils 
    - save files 
    - load files 
    - pipeline creation used in testing data_processing.py 



# workflow 
- load in the pipeline 
- drop ID column for training 
- drop Id column for prediction 

    
### notes: 
- the reason why this is a classification problem rather than regression is that the regression model does not have enough samples to train on. it is possibly (?) easier to train on a classifyer rather than a regressor. 