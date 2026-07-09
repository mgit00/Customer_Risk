'''
this contains the kmeans model and fnn model 
'''

from sklearn.cluster import KMeans
import torch.nn as nn 


class KMeansModel: 
  '''
  first make kmeans model with the chosen k 
  this will be trained to get centroid locations
  then go through each instance and get distances form the instance to the centroid location
  '''
  def __init__(self, args): 
    self.args = args 
    self.model= None # initialize 

  def train_model(self, data): # returns trained model 
    k = self.args.chosen_k_value

    kmeans= KMeans(n_clusters=k , random_state=1)
    cluster_labels= kmeans.fit_predict(data )

    self.model= kmeans # save to init 
    return kmeans

  def predict(self, data): 
    pred = self.model.predict(data) 
    return pred 



class FNN(nn.Module): 
  def __init__(self): 
    FNN.sup


  

