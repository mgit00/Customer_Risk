'''
data processing 
calls training 
calls predicting 
'''

import pandas as pd 
import numpy as np 

from model import KMeansModel
from train import Train 

from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split

import torch 
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

class ModelHelper: 
  def __init__(self, args ): 
    self.args= args 
    self.kmeansmodel= None # initalized later in training()
    self.train_instance= None # initalized later in training()



  def get_data(self, f): 
    '''
    get the data from the csv files 
    for training loop 
    convert to tensors 
    '''
    data= pd.read_csv(f )
    data.drop(columns= ['ID'], axis=1 , inplace = True )

    return data 


  def extract_features(self, data:np.array, centroids) -> pd.DataFrame: 
    '''
    for the centroid locations, go through data and get euclidean distance
    ''' 

    # for i in range(self.args.chosen_k_value): 
    #   distance = np.linalg.norm(data- centroids[i], axis=1 )
    distances_df = pairwise_distances(data, centroids, metric= 'euclidean')

    return distances_df 


  def kmeans_training(self, datax, full_data):
    '''
    get kmeans clusters, and make centroids into features 
    join the data to the cluster-distance features 
    '''

    # first make kmeans model with the chosen k 
    # this will be trained to get centroid locations
    # this can be done with the larger dataset application_file to improve model performance 
    # and prevent overfitting 
    self.kmeansmodel= KMeansModel(self.args)
    self.kmeansmodel.train_model(full_data)

    # now get the predicitons for the application_risk dataset 
    pred = self.kmeansmodel.predict(datax)
    return pred 

  def loading_data(self, x, y): # -> dataloader objects 
    '''
    convert data to tensors 
    place data onto gpu if available 

    split data (shuffle in traintestsplit and dataloader to ensure
     unbiased split and robust model training from epoch to epoch )
    make into tensor dataset 
    make into dataloader object 
    '''
    x= torch.tensor(x)
    y= torch.tensor(y)

    device= torch.device('cuda' if torch.cuda.is_available() else 'gpu')
    x= x.to(device) # dataset is relatively small so can place it entirely on gpu now 
    y= y.to(device)

    # split into train test chunks 
    xtrain, xtest, ytrain, ytest= train_test_split(x, y, test_size=.15, shuffle=True, random_state=1 )


    # make into tensor dataset 
    train = TensorDataset(xtrain, ytrain)
    test= TensorDataset(xtest, ytest)

    # make into 
    train_loader= DataLoader(train, batch_size=self.args.batch_size, shuffle= True)
    test_loader= DataLoader(test, batch_size=1, shuffle= True )

    return train_loader, test_loader 




  def training(self): 
    '''
    get kmeans clusters, and make centroids into features 
    join the data to the cluster-distance features 
    transform data into tensors 
    apply FNN from train.py to get classifications 
    backpropogate losses 

    calls train.py 
    ''' 
    full_data= self.get_data(self.args.application_file)
    data= self.get_data(self.args.application_risk_file)
    datax, datay = data.iloc[:, :-1], data.iloc[:,-1]

    pred= self.kmeans_training(datax, full_data)
    
    # go through each instance and get distances from the instance to the centroid location 
    distances_df= self.extract_features(datax, self.kmeansmodel.model.cluster_centers_)
    datax_distances= pd.concat([datax, distances_df], axis=1 )
    # the datax_distances df now has the customer application risk and distance from clusters as features 
    
    # convert to tensors as needed and continue onto FNN 
    train_loader, test_loader= self.loading_data(datax_distances, datay)


    # use the data to pass to the FNN now 
    # begin training loop 
    self.train_instance= Train(self.args)
    for epoch in range(self.args.epochs): 

      print(epoch)


    raise Exception('here')



    



  # def predicting(self): 
  #   '''
  #   calls predict.py 
  #   '''
