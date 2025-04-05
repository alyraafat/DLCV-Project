from torch import nn

class Second_Model(nn.Module):
    def __init__(self):
        super(Second_Model, self).__init__()
        
        # Layer 1 , 32 filters each is 3x3, ReLU activation, MaxPooling
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # Layer 2 , 64 filters each is 3x3, ReLU activation, MaxPooling
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Layer 3 , 64 filters each is 3x3 , ReLU activation, MaxPooling
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # Layer 4 , 32 filters each is 5x5, ReLU activation, MaxPooling
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=0)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        # Layer 5 , 16 filters each is 7x7, ReLU activation, MaxPooling
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, padding=0)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=2)
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Densy layer with 128 neurons and sigmoid activation
        self.fc1 = nn.LazyLinear(128)
        self.sigmoid = nn.Sigmoid()        

        # Output layer with 4 neurons and softmax activation
        self.fc2 = nn.Linear(128, 4)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Layer 1 forward pass
        x = self.pool1(self.relu1(self.conv1(x)))

        # Layer 2 forward pass
        x = self.pool2(self.relu2(self.conv2(x)))

        # Layer 3 forward pass
        x = self.pool3(self.relu3(self.conv3(x)))

        # Layer 4 forward pass
        x = self.pool4(self.relu4(self.conv4(x)))

        # Layer 5 forward pass
        x = self.pool5(self.relu5(self.conv5(x)))
        
        # Flatten the output
        x = self.flatten(x)

        # Fully connected layer with sigmoid activation
        x = self.sigmoid(self.fc1(x))

        # Output layer with softmax activation
        x = self.softmax(self.fc2(x))
        return x