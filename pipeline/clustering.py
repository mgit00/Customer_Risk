import pandas as pd 

from models import KMeansModel

class Clustering: 
    '''
    this class drives the creation of kmeans for multiple k 
    '''
    def __init__(self, args): 
        self.args = args 
    
    def process_data(self, data) -> pd.DataFrame: 
        # dont do inplace change, as this might affect pipeline upstream 
        new_data= data.drop(columns= ['ID', 'risk_level'], axis =1 , inplace= False)
        return new_data 
    
    def get_intertias(self, data)-> list: # returns list of intertias  
        # process data first 
        data= self.process_data(data)

        kstart, kend = self.args.k_values 
        intertias = []

        for k in range(kstart, kend+1): # inclusive 
            # instantate model 
            kmeans= KMeansModel(k)
            cluster_labels= kmeans.forward(data)
            intertias.append(kmeans.model.inertia_)

        print('intertias for kmeans are: ', intertias)
        return intertias


    