import torch.nn as nn 

class FNN(nn.Module): 
    '''
    the most simple model 
    '''
    def __init__(self, args, input_size, hidden_sizes, output_size): 
        super().__init__()

        self.layers= nn.Sequential()

        self.layers.append(nn.Linear(input_size, hidden_sizes))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_sizes, output_size))

        print('FNN model is created with layer sizes: ', input_size, hidden_sizes, output_size)
        
    def forward(self, x): 
        # return output 
        out = self.layers(x)
        return out 