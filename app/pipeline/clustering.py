import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

from .utils import Utils 
from .plotting import Plotting

import logging 

class Clustering: 
    '''
    this class drives the creation of kmeans for multiple k 
    '''
    def __init__(self, args): 
        self.args = args 
        self.logger = logging.getLogger(self.args.logger_name)

        self.plotting = Plotting(self.args)

    def driver(self): 
        data= self.process_data() # full data 
        intertias = self.get_intertias(data)
        self.plotting.kmeans_elbow_curve(intertias) # this saves the image 
        return 


    def process_data(self) -> pd.DataFrame: 
        # read data 
        data = Utils.read_data(self.args.data_filename)
        self.logger.debug('data is loaded for clustering ')

        x,y = data.iloc[:, :-1], data.iloc[:, -1:] # df, and df (not series, makes mode code more complicated to switch between classificaiton and regression later on) 

        trainx, testx, trainy, testy = train_test_split(x, y, test_size=.3, random_state= 1, shuffle = True)
            # random state makes it replicable 
            # batching not done here, done in dataloader 
            # shuffle rows 

        self.logger.info('data split is done')
        
        return trainx 
    
    def get_intertias(self, data)-> list: # returns list of intertias  
        ''' 
        for all of the k values, it creates the kmeans model 
        returns the itertias 
        '''
        # note: data has been processed already  

        kstart, kend = self.args.k_values 
        intertias = []

        for k in range(kstart, kend+1): # inclusive 
            # instantate model 
            kmeans= KMeans(n_clusters= k, random_state= 1 )
            cluster_labels= kmeans.fit_predict(data)
            intertias.append(kmeans.inertia_)

        self.logger.info('intertias for kmeans are: ' + str(intertias))
        return intertias


    