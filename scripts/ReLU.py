import numpy as np

class ReLU:
    """
    Implements the ReLU activation function.
    """

    def __init__(self):
        pass

    def forward(self, x):
        """
        Applies the ReLU activation function.
        """
        return np.maximum(0, x)
