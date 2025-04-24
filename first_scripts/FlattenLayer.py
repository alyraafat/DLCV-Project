import numpy as np

class FlattenLayer:
    """
    A class that represents a Flatten layer in a neural network.
    This layer flattens the input tensor into a 1D tensor.
    """

    def __init__(self):
        """
        Initializes the FlattenLayer instance.
        """
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Flattens the input tensor.

        Args:
            x (np.ndarray): Input tensor of shape (height, width, channels).

        Returns:
            np.ndarray: Flattened tensor of shape (1, height * width * channels).
        """
        flattened = x.reshape(1, -1)
        return flattened