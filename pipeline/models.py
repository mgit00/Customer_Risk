import torch.nn as nn 

from sklearn.cluster import KMeans
class KMeansModel: 
    def __init__(self, kvalue): 
        self.model= KMeans(n_clusters= kvalue, random_state= 1 ) # random state for reproducability 
    
    def forward(self, data):
        cluster_labels= self.model.fit_predict(data)
        return cluster_labels


    

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
    


class KMeansFNN(nn.Module): 
    '''
    model which takes the kmeans_classifications and adds it to the data
    then trains the fnn model
    '''
    def __init__(self, k_value, input_size, hidden_sizes, output_size): 
        super().__init__()

        self.fnn= FNN(input_size, hidden_sizes, output_size)

        
        pass 

class AutoencoderKMeansFNN(nn.Module): 
    def __init__(self): 
        pass 