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
- how to deal with error with loading in model 
- have to lead in with loaded model when intiated 
- get the modelstate updated dict before passedinto train 
- same for optimizer 

- may have to create a new class to laod in model
- this work has to be done before train is called 

- may need a class to deal with setting up mode, loss func and optimizer 


# ignored, but kept for notes: 
- fnn and slp 
        load data 
        make dataloader
        train.py -> model.py

    - kmeans + fnn 
        # load data 
        # train kmeans on xtrain --> xtrain+clusters 
        # make dataloader with xtrain+clusters 
        # train.py -> model.py 
    
        # or 
        # do below, makes code easier and more modular 
            # easy to build upon later on 

        load data 
        train kmeans on xtrain -> trained kmeans model 
        make dataloader with xtrain
        train.py -> model.py 
            model gets xtrain data, predicts on trained kmeans, passes to nn layers 
    
    - autoencoder + kmeans + fnn 
        load data 
        make dataloader 
        train.py -> model.py 
            autoencoder 
        
        predict on entire xtrain data (to get compressed embeddings)
            -> use to train kmeans model 

        return to xtrain data! 
        make dataloader 
        train.py -> model.py 
            fnn (pred on AE for embeddings, pred on kmeans for clusters, pred on fnn )


    RETHINKING APPROACH: 
    this means that pred will be called for number of epochs * data divided by num batches 
    which means that it will be computationally expensive 
    instead seperate out pipelines for each model 
        so you dont have to predict on trained autoencoder and kemans model for each train epoch of the fnn 
    
    however, for the predicting portion of the pipeline, i will have to do this 

    so, for training, do: 
    - fnn and slp 
        load data 
        make dataloader
        train.py -> model.py

    - kmeans + fnn 
        load data 
        train kmeans on xtrain --> xtrain+clusters 
        make dataloader with xtrain+clusters 
        train.py -> model.py 

    - autoencoder + kmeans + fnn 
        load data 
        make dataloader 
        train.py -> model.py 
            autoencoder (training)
        
        predict xtrain with trained autoencoder (full data, to get compressed embeddings) -> xtrain_embeddings
            -> use to train kmeans model 
        kmeans model training -> xtrain_embeddings+clusters 

        make dataloader with xtrain_embeddings+clusters 
        train.py -> model.py 
            fnn 

    
    for predicting do: 
        load data 
        make dataloader 
        train.py -> model.py 
            call predict() for each model to call on the different steps for each model 
    

    better workflow done below :) 

