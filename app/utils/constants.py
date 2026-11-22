'''
this file contains constant information
''' 

from enum import Enum 

class ModelType(str, Enum): 
    '''
    gets name and value for various models to be implemented 
    there are 4 models to be trained 
    '''
    slp  = 'slp_model'
    fnn = 'fnn_model'
    kmeans_fnn = 'kmeans_fnn_model'
    autoencoder_kmeans_fnn = 'autoencoder_kmeans_fnn_model'


class ConstantFilenames(str, Enum): # changed from basemodel, keep these values static  
    '''
    class to deal with data processing pipeline filenames 
        there is no a specific configuration for the pipeline, so it is ok to use constant filenames  
    
    read in: 
        dataset/  

    save to: 
        saved_data/ 
            data_processing/ 
            clusters/ 
            trained_models/ 
                slp/ 
                fnn/ 
                ... 
    '''
    ### data processing 
    # customer credit application file - raw data 
    application_record = 'dataset/application_record.csv'
    # customer history - raw data 
    credit_record = 'dataset/credit_record.csv' 


    ### processed data 
    # saved pipeline
    data_pipeline = 'data_saved/data_processing/application_record_pipeline.pkl'
    # saved processed data file 
    data_filename = 'data_saved/data_processing/data.csv'


    ### for clustering model saving: 
    elbow_curve_for_kmeans = 'data_saved/clusters/elbow_curve_for_kmeans.png' # plot 


    ### saved trained models 
        # just 1 file at a time, just needed to make the parent folders
        # cant do _name_, ValueError: _names_ are reserved for future Enum use 
    _slp= f'data_saved/trained_models/{ModelType.slp.name}/temp.pt'
    _fnn= f'data_saved/trained_models/{ModelType.fnn.name}/temp.pt'
    _kmeans_fnn= f'data_saved/trained_models/{ModelType.kmeans_fnn.name}/temp.pt'
    _autoencoder_kmeans_fnn= f'data_saved/trained_models/{ModelType.autoencoder_kmeans_fnn.name}/temp.pt'

