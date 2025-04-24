from torch import nn 


class CNNBlock(nn.Module):
    def __init__(self):
        '''
        Initializes the CNN block with multiple convolutional layers, ReLU activations, max pooling, and batch normalization.
        '''
        super(CNNBlock, self).__init__()

        # Convolution 1
        self.conv1 = nn.LazyConv2d(32, kernel_size=3)
        self.relu1 = nn.ReLU()

        # Convolution 2
        self.conv2 = nn.LazyConv2d(64, kernel_size=3)
        self.relu2 = nn.ReLU()

        # Convolution 3
        self.conv3 = nn.LazyConv2d(64, kernel_size=3)
        self.relu3 = nn.ReLU()

        # Convolution 4
        self.conv4 = nn.LazyConv2d(32, kernel_size=5)
        self.relu4 = nn.ReLU()

        # Convolution 5
        self.conv5 = nn.LazyConv2d(16, kernel_size=7)
        self.relu5 = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.batchnorm = nn.BatchNorm2d(16)

    def forward(self, x):
        '''
        Forward pass through the CNN block.
        
        Args:
            x (torch.Tensor): Input data (N, C, H, W).
            
        Returns:
            torch.Tensor: Convolved output (N, C', H', W').
        '''
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.maxpool(x)
        x = self.batchnorm(x)
        return x
