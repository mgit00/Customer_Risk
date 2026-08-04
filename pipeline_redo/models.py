import torch 
import torch.nn as nn 

class SLP(nn.Module): 
    def __init__(self, input_size, output_size ): 
        super().__init__()
        self.layer = nn.Linear(input_size, output_size)


    def forward(self, data): 
        res = self.layer(data)
        return res  