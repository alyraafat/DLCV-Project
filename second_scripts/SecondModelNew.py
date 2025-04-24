from torch import nn 
from .CNNBlock import CNNBlock
import torch

class SecondModel2(nn.Module):
    def __init__(self, num_classes: int=5):
        super(SecondModel2, self).__init__()
        
        blocks = []
        for _ in range(3):
            blocks.append(CNNBlock())

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
    

if __name__ == "__main__":
    model = SecondModel2()
    print(model)
    # Test the model with a random input
    x = torch.randn(1, 3, 512, 512)  # Example input (batch_size=1, channels=3, height=32, width=32)
    output = model.predict_proba(x)
    print("Output shape:", output.shape)  # Should be (1, num_classes)
