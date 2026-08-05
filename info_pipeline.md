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
- argument_parser 
    - argparse 
- driver
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


---------- 
- driver.py 
    - calls: 
        - data_loading 
        - training 
        - models 
        - model saving 
        - plotting 

- models.py 
    - 1 model class for each model 

- model_helper.py 
    - load in the data and batch/process/ect as needed 


### workflow for model building 
there should be 3 models: 
- knn 
- kmeans + knn 
- autoencoders + knn 

the requirements are: 
- each model requires its own model class in models.py 
- train.py should call 1 of each model as needed 
    - as a result, the differentiation between the models doesn't have to be done in pipeline.py; but instead can be done in training.py 
    - or for more flexibility, it can be differentiated early in pipeline.py but integrated well later on st. the calls in pipeline.py don't have a large effect 



# TBD: workflow 
- load in the pipeline 
- drop ID column for training 
- drop Id column for prediction 
---
- argument parser 
- driver 
- model helper 
    - load data  
    - training 
        - calls train.py for training loop 
        - plots training loss 


- build out taking in trained model and continue training 


---- 
how to deal with kmeans algorithm? 
the general nn workflow is done well 
inorder to deal with kmeans, a k has to be chosen first 
    - to choose a k, the elbow curve should be plotted first 

- create a seperate pipeline to deal with keamns algorithm 

    
### notes: 
- the reason why this is a classification problem rather than regression is that the regression model does not have enough samples to train on. it is possibly (?) easier to train on a classifyer rather than a regressor. 


### autoencoder model 
for the autoencoder kmeans fnn model , the last training loop which deals with the fnn model has to be adjusted a bit 
because the model also contains the autoencoder layers, backpropating the loss gradients will also lead to the autoencoder's layers being updated. this shouldn't happen ideally bcause the autoencoder should already be fully trained. 

to prevent the gradients from being backpropogated, there are 3 things that can be done: 
1) freeze the params (anytime before training loop?)
2) detatch the output tensor (in the forward() function )
3) exclude the autoencoder params from the optimizer 


### now ... 
- need to save model state or atleast get it back to start plotting 
- return 
    - args 
    - losses 
    - losses test 
    - accuracy 
    - accuracy test 
    - f1 score 
    - f1 score test 

- rework how to deal with metrics 
