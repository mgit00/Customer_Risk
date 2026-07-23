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
    - processes the 
    
- utils 
    - save files 
    - load files 



# workflow 
- <input type="checkbox" > pipeline.py 
    - <input type="checkbox" checked> set up argparse  
    - <input type="checkbox" > pipleine workflow class 
        - <input type="checkbox" > data processing of both dfs 
- <input type="checkbox" > utils.py 
    - <input type="checkbox" checked> save /read df 
- <input type="checkbox"> data_processing.py 
    - <input type="checkbox" checked> convert credit_record -> customer_risk
    - <input type="checkbox" > convert application_record -> application_processing 

    
### notes: 
- the reason why this is a classification problem rather than regression is that the regression model does not have enough samples to train on. it is possibly (?) easier to train on a classifyer rather than a regressor. 