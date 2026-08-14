# Customer Risk
This project uses credit card application and bank record data from https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction. 

With this data, i will create an end-to-end pipeline which classifies the risk level of approving a credit card application for a customer. 

This can be integrated into a much larger application which can ultimatly decide whether or not to apporve a credit card application. This larger system will take into account the national economy, local neightborhood of the applicant, and the company's overall need to expand into a larger customer base (which requires more approvals). 

This project does the following: 
- Docker containerization to ensure environment consistency 
- Cloud deployment in Amazon ECS (Elastic Container Service) 
- FastAPI backend provides endpoints classifying the risk of approving a customer's credit card applicatoin classification
- FastAPI backend provides endpoints for model training and hyperparameter tuning, allowing for a dynamic-scalable model which can be retrained as customer records grow.
- Git for easy version control 
- Modular scikit-learn pipelines for preprocessing. 
- Advanced Neural Networks which integrate clustering algorithms: SLP, FNN, KMeans clustering algorithm with FNN, Autoencoder NN combined witH KMeans and FNN. 

## Overview

This project was designed to demonstrate a complete machine learning workflow, from data preprocessing and model training to API-based inference.

The system supports two prediction tasks:

* Classification — predicts one of five customer risk levels.
* Regression — predicts a continuous numerical risk score.

A minimum of 8 machine learning models are implemented and can be selected through the training interface. More model versions can be created so that the best model can be utilized.  

The code focuses on creating clean, modular and reusable code, allowing individual preprocessing steps, models, and training configurations to be changed without requiring major changes to the rest of the application.



## Key Features

* 5-class credit risk classification
* Credit risk regression
* Four machine learning algorithms
* Reusable scikit-learn preprocessing pipelines
* FastAPI training API
* FastAPI prediction API
* Automated training-loss visualization
* Configurable model training using `argparse`, accessible through command line or FastAPI
* Ability to train individual models or multiple models
* Model evaluation and comparison with multiple evaluation metrics
* Elbow curve analysis for different `k` values
* Modular architecture separating data, preprocessing, training, and inference
* Saved trained pipelines for consistent prediction
* Reproducible training workflow


## Machine Learning Tasks

To create the classification and regression models, the data is processed from its raw format. Risk scores for classification and regression are computed by me personally rather than relying on another algorithm to do so. I did this so that there would be no data leakage from the credit application data. 

### Classification

The classification system predicts the customer's application risk using five risk levels.

- No risk 
- Low risk 
- Medium-Low risk 
- Medium-High risk 
- High risk 

The exact interpretation of each level is determined by the dataset and target definition.

The classification pipeline handles preprocessing before passing the transformed features to the selected classification model.

### Regression

The regression workflow predicts a continuous risk value rather than assigning the customer to one of five discrete categories.

This is done with tweaking a small portion of the preprocessing infrastructure, showing how the pipeline is easily adjusted without interupting the rest of the model workflow. 

## FastAPI 
FastAPI is used to interact with the pipeline. 
api_framework/fastapi_setup.py has endpoint connection information. 
