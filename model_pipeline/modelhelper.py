'''
data processing 
calls training 
calls predicting 
'''

import pandas as pd 

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
    
  
  # def clustering(self, data:pd.DataFrame): 
  #   '''
  #   with the pandas df, get the right k value for k means 
  #   apply the clustering model 
    # '''



  # def training(self): 
  #   '''
  #   calls train.py 
  #   ''' 

  # def predicting(self): 
  #   '''
  #   calls predict.py 
  #   '''