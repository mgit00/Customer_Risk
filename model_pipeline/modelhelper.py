import pickle 

import pandas as pd 
import numpy as np
from scipy.spatial import transform 

from model import KMeansModel
from train import Train 
from plotting import TrainingPlotting

from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler

import torch 
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader




class ModelHelper: 
  '''
  data processing 
  calls training 
  calls predicting 
  '''
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

    # scale the distances to -1 to 1 
    # scaler= RobustScaler().set_output(transform= 'pandas') # returns np array
    # distances_df= scaler.fit_transform(distances_df)
    scaler= MinMaxScaler(feature_range= (-1, 1)).set_output(transform= 'pandas')
    distances_df = scaler.fit_transform(distances_df)

    return distances_df 


  def kmeans_training(self, datax, full_data):
    '''
    get kmeans clusters, and make centroids into features 
    join the data to the cluster-distance features

    - do work with saving or loading model  
    '''

    if self.args.load_model_kmeans: 
      print('Loading in KMeans model from ', self.args.load_model_kmeans)
      self.kmeansmodel= pickle.load(open(self.args.load_model_kmeans, 'rb'))
    else: 
      # first make kmeans model with the chosen k 
      # this will be trained to get centroid locations
      # this can be done with the larger dataset application_file to improve model performance 
      # and prevent overfitting 
      print('Training, not loading in KMeans model')
      self.kmeansmodel= KMeansModel(self.args)
      self.kmeansmodel.train_model(full_data)

    # now get the predicitons for the application_risk dataset 
    pred = self.kmeansmodel.predict(datax)

    # save model if needed: 
    if self.args.save_model_kmeans: 
      print('Saving KMeans model to ', self.args.save_model_kmeans)
      pickle.dump(self.kmeansmodel, open(self.args.save_model_kmeans, 'wb'))
    
    
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
    x= torch.tensor(x.values, dtype=torch.float32) # values to convert to np array 
    y= torch.tensor(y.values, dtype= torch.float32)

    self.args.device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device Used:', self.args.device)
    x= x.to(self.args.device) # dataset is relatively small so can place it entirely on gpu now 
    y= y.to(self.args.device) # reassign to ensure cuda is used, and we aren't just creating a copy on cupu 

    # split into train test chunks 
    xtrain, xtest, ytrain, ytest= train_test_split(x, y, test_size=.15, shuffle=True, random_state=1 )


    # make into tensor dataset 
    train = TensorDataset(xtrain, ytrain)
    test= TensorDataset(xtest, ytest)

    # make into 
    train_loader= DataLoader(train, batch_size=self.args.batch_size, shuffle= True)
    test_loader= DataLoader(test, batch_size=self.args.batch_size, shuffle= True )

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
    datax, datay = data.iloc[:, :-1], data.iloc[:,-1].to_frame()

    pred= self.kmeans_training(datax, full_data)
    print('K Means Portion Done')

    # go through each instance and get distances from the instance to the centroid location 
    distances_df= self.extract_features(datax, self.kmeansmodel.model.cluster_centers_)
    distances_df= pd.DataFrame(distances_df)
    datax_distances= pd.concat([datax, distances_df], axis=1 )
    # the datax_distances df now has the customer application risk and distance from clusters as features 
    # save data
    datax_distances.to_csv('dataset_redo/kmeans_customer_application_risk.csv', index= False )
    print('Final Dataset w/ KMeans saved to: dataset_redo/kmeans_customer_application_risk.csv ')

    # convert to tensors as needed and continue onto FNN 
    self.args.input_size = datax_distances.shape[1] # number of features 
    self.args.output_size = 1 # series 
    train_loader, test_loader= self.loading_data(datax_distances, datay)


    # use the data to pass to the FNN now 
    # begin training loop 
    print('Beginning FNN Portion')
    self.train_instance= Train(self.args)
    model_state= self.train_instance.training_loop(train_loader, test_loader)

    # plot the KNN training 
    trainingplotting= TrainingPlotting(self.args)
    trainingplotting.plotting(model_state['epochs_completed'], 
                              model_state['train_loss_data'], 
                              model_state['test_loss_data'], )


    print('KNN Training Done')
    
    return # None 



    



  # def predicting(self): 
  #   '''
  #   calls predict.py 
  #   '''
