'''
data processing 
calls training 
calls predicting 
'''

import pandas as pd 
import numpy as np 

from model import KMeansModel

from sklearn.metrics import pairwise_distances

class ModelHelper: 
  def __init__(self, args ): 
    self.args= args 


  def get_data(self, f): 
    '''
    get the data from the csv files 
    for training loop 
    convert to tensors 
    '''
    data= pd.read_csv(f )
    data.drop(columns= ['ID'], axis=1 , inplace = True )

    return data 

  def tensor_data(self, f): 
    # pandas df 
    data= self.get_data(f)


  def extract_features(self, data:np.array, centroids) -> pd.DataFrame: 
    # for the centroid locations, go through data and get euclidean distance 

    # for i in range(self.args.chosen_k_value): 
    #   distance = np.linalg.norm(data- centroids[i], axis=1 )
    distances_df = pairwise_distances(data, centroids, metric= 'euclidean')

    return distances_df 




  def training(self): 
    '''
    get kmeans clusters, and make centroids into features 
    join the data to the cluster-distance features 
    transform data into tensors 
    apply FNN from train.py to get classifications 
    backpropogate losses 

    calls train.py 
    ''' 
    data= self.get_data(self.args.application_file)
    data= data.values 

    # first make kmeans model with the chosen k 
    # this will be trained to get centroid locations
    kmeansmodel= KMeansModel(self.args)
    model = kmeansmodel.train_model(data)

    # then go through each instance and get distances form the instance to the centroid location 
    distances_df= self.extract_features(data, model.cluster_centers_)
    data_distances= np.concat([data, distances_df], axis=1 )
    data_distances = pd.DataFrame(data_distances)

    # now join the distances to the risk based on ID 
    ids= pd.read_csv(self.args.application_file).ID
    full_data= ids.join(data_distances) # join based on index 
    
    risk= pd.read_csv(self.args.risk_file)

    joined= here

    raise Exception('here')


    



  # def predicting(self): 
  #   '''
  #   calls predict.py 
  #   '''
