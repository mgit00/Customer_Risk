'''
data processing 
calls training 
calls predicting 
'''

import pandas as pd 
import numpy as np 

from model import KMeansModel

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
    # for the centroid locations, go through data and 
    print(centroids)
    data.




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

    raise Exception('here')


    



  # def predicting(self): 
  #   '''
  #   calls predict.py 
  #   '''