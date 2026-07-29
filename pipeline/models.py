import torch.nn as nn 

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
    def __init__(self): 
        pass 

class AutoencoderKMeansFNN(nn.Module): 
    def __init__(self): 
        pass 