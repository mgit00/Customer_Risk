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



  # def training(self): 
  #   '''
  #   calls train.py 
  #   ''' 

  # def predicting(self): 
  #   '''
  #   calls predict.py 
  #   '''