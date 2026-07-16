import torch.nn as nn 
from sklearn.cluster import KMeans

# '''
# this contains the kmeans model and fnn model 
# '''


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
  def __init__(self, args, input_size, hidden_sizes, output_size): 
    super().__init__()

    '''
    make model of 1 layer for trial 
    '''
    self.args = args 

    if not self.args.hidden_sizes: # no layers specified, default to simple model here 
      self.layers= nn.Sequential(
        nn.Linear(input_size, 32), 
        nn.ReLU(), 
        nn.Linear(32, output_size)
      )
    else: 
      hiddens= self.args.hidden_sizes
      self.layers= nn.Sequential()

      self.layers.append(nn.Linear(input_size, hiddens[0]))
      self.layers.append(nn.BatchNorm1d(hiddens[0])) # because of 0,1 scaling 
      self.layers.append(nn.LeakyReLU())
      for i in range(len(hiddens)-1): 
        self.layers.append(nn.Linear(hiddens[i], hiddens[i+1]))
        self.layers.append(nn.BatchNorm1d(hiddens[i+1]))
        self.layers.append(nn.LeakyReLU()) # so model doesn't collapse into linear model 
      self.layers.append(nn.Flatten())
      self.layers.append(nn.Linear(hiddens[-1], output_size))


    print('FNN model is: ' , self.layers)


  def forward(self, data): 
    res= self.layers(data)

    return res 


  

