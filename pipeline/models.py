import torch.nn as nn 

from sklearn.cluster import KMeans


class SLP(nn.Module): 
    '''
    single layer perceptron 
    used for checking if pipeline is set up correctly
    '''
    def __init__(self, input_size, output_size): 
        super().__init__()
        self.linear = nn.Linear(input_size, output_size) 

        print('SLP model is created with layer sizes: ', input_size, output_size)

    def forward(self, data): 
        return self.linear(data)



class FNN(nn.Module): 
    '''
    the most simple model 
    '''
    def __init__(self, input_size, hidden_sizes: list, output_size): 
        super().__init__()

        self.layers= nn.Sequential()

        # hidden layers is a list 
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(p=0.3))
        for i in range(len(hidden_sizes)-1) : 
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=0.3))
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

        print('FNN model is created with layer sizes: ', input_size, hidden_sizes, output_size)
        
    def forward(self, x): 
        # return output 
        out = self.layers(x)
        return out 
    

class KMeansModel: 
    '''
    called in model_helper to train and save model 
    the models is loaded in later on to use in train.py predict
    '''
    def __init__(self): 
        # model initialize 
        self.model = KMeans(n_clusters= self.args.k_value, random_state=1 )
         
    def train(self, data): 
        # train the model 
        self.model.fit(data)
        pass 

    def predict_data(self,data_batch): 
        # cluster_distances = # get distances to each label
        pass 
        # return databatch with the centroid distances 

class KMeansFNN(nn.Module): 
    '''
    model which takes the kmeans_classifications and adds it to the data
    then trains the fnn model

    the trained kmeans model is passed in 
    '''
    def __init__(self, k_value, input_size, hidden_sizes, output_size): 
        super().__init__()

        self.fnn= FNN(input_size, hidden_sizes, output_size)


        pass 

class AutoencoderKMeansFNN(nn.Module): 
    '''
    model which takes the autoencoder applied kmeans centroid distances and adds it to the data 
    trained autoencoder reference is passed in 
    trained kmeans model reference is passed in 
    '''
    def __init__(self): 
        pass 