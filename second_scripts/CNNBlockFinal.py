from torch import nn 


class CNNBlock2(nn.Module):
    def __init__(self, num_filters: int=32, kernel_size: int=3):
        '''
        Initializes the CNN block with multiple convolutional layers, ReLU activations, max pooling, and batch normalization.
        '''
        super(CNNBlock2, self).__init__()

        # Convolution 1
        self.conv1 = nn.LazyConv2d(num_filters, kernel_size=kernel_size)

        # Convolution 2
        self.conv2 = nn.LazyConv2d(num_filters, kernel_size=kernel_size)

        # Convolution 3
        self.conv3 = nn.LazyConv2d(num_filters, kernel_size=kernel_size)
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.batchnorm = nn.LazyBatchNorm2d()

    def forward(self, x):
        '''
        Forward pass through the CNN block.
        
        Args:
            x (torch.Tensor): Input data (N, C, H, W).
            
        Returns:
            torch.Tensor: Convolved output (N, C', H', W').
        '''
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.batchnorm(x)
        return x
