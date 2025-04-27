from torch import nn 
from .CNNBlockFinal import CNNBlock2
import torch
from typing import List

class SecondModel3(nn.Module):
    def __init__(self, num_classes: int=5, num_filters: List[int]=[32,64,64,32,16], kernel_size: List[int]=[3,3,3,5,7]):
        super(SecondModel3, self).__init__()
        
        blocks = []
        for i in range(5):
            curr_filter = num_filters[i]
            curr_kernel = kernel_size[i]
            blocks.append(CNNBlock2(num_filters=curr_filter, kernel_size=curr_kernel))

        self.cnn_blocks = nn.Sequential(*blocks)
        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(32)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.LazyLinear(num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        '''
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input data (N, C, H, W).
            
        Returns:
            torch.Tensor: Output (N, num_classes).
        '''
        x = self.cnn_blocks(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        # x = self.softmax(x)
        return x
    
    def predict_proba(self, x):
        '''
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input data (N, C, H, W).
            
        Returns:
            torch.Tensor: Output (N, num_classes).
        '''
        x = self.forward(x)
        x = self.softmax(x)
        return x
    
    def predict(self, x):
        '''
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input data (N, C, H, W).
            
        Returns:
            torch.Tensor: Output (N, num_classes).
        '''
        x = self.predict_proba(x)
        return x.argmax(dim=1)

