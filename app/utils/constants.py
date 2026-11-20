'''
this file contains constant information
''' 

from enum import Enum 

class ModelType(str, Enum): 
    '''
    gets name and value for various models to be implemented 
    there are 4 models to be trained 
    '''
    slp  = 'slp'
    fnn = 'fnn'
    kmeans_hybrid = 'kmeans+fnn'
    autoencoder_kmeans_hybrid = 'autoencoder+kmeans+fnn'

