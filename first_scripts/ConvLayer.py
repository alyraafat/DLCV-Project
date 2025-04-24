from typing import Tuple, List
import numpy as np

class ConvLayer:
    # def __init__(self, num_filters: int, kernel_size: Tuple[int, int, int]):
    #     '''
    #     Initializes a convolutional layer with the given number of filters and input shape.
        
    #     Args:
    #         num_filters (int): Number of filters in the convolutional layer.
    #         kernel_size (Tuple[int, int, int]): Size of the convolutional kernel (height, width, channels).
    #     '''
    #     self.num_filters = num_filters
    #     self.kernel_size = kernel_size
    #     self.filters = np.random.randn(num_filters, *kernel_size) * 0.01  # Initialize filters with small random values

    def __init__(self, num_filters: int, kernel_size: Tuple[int, int], filter_weights: List[np.ndarray] = None, fast_convolution: bool = True):
        '''
        Initializes a convolutional layer with the given number of filters, kernel size, and filter weights.

        Args:
            num_filters (int): Number of filters in the convolutional layer.
            kernel_size (Tuple[int, int]): Size of the convolutional kernel.
            filter_weights (List[np.ndarray]): Weights for the filters.
        '''
        self.num_filters = num_filters
        self.fast_convolution = fast_convolution
        self.kernel_size = kernel_size
        if filter_weights is None:
            self._random_init(num_filters, kernel_size)
        else:
            assert len(filter_weights) == num_filters, "Number of filter weights must match the number of filters"
            self._set_weights(filter_weights)

    def _random_init(self, num_filters: int, kernel_size: Tuple[int, int]):
        '''
        Initializes the filters with small random values.
        
        Args:
            num_filters (int): Number of filters.
            kernel_size (Tuple[int, int]): Size of the convolutional kernel.
        
        Returns:
            np.ndarray: Initialized filters.
        '''
        self.filters = np.random.randn(num_filters, *kernel_size) * 0.01
    
    def _set_weights(self, filter_weights: List[np.ndarray]) -> None:
        '''
        Sets the weights for the filters.
        
        Args:
            filter_weights (List[np.ndarray]): Weights for the filters.
        '''
        self.filters = np.array(filter_weights).reshape(self.num_filters, *self.kernel_size)



    def _convolve(self, input_data: np.ndarray) -> np.ndarray:
        '''
        Applies the convolution operation on the input data using the filters.
        
        Args:
            input_data (np.ndarray): Input data (H, W, C).
        
        Returns:
            np.ndarray: Convolved output (H', W', num_filters).
        '''
        input_h, input_w, channels = input_data.shape
        kernel_h, kernel_w = self.filters.shape[1:3]
        output_h = input_h - kernel_h + 1
        output_w = input_w - kernel_w + 1
        output = np.zeros((output_h, output_w, self.num_filters))
        for f in range(self.num_filters):
            for h in range(output_h):
                for w in range(output_w):
                    h_start = h
                    h_end = h_start + kernel_h
                    w_start = w
                    w_end = w_start + kernel_w
                    region = input_data[h_start:h_end, w_start:w_end, :]
                    # print(f"Region shape: {region.shape}, Filter shape: {self.filters[f].shape}")
                    output[h, w, f] = np.dot(region.flatten(), self.filters[f].flatten())
        return output
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        '''
        Forward pass through the convolutional layer.
        
        Args:
            input_data (np.ndarray): Input data (H, W, C).
        
        Returns:
            np.ndarray: Convolved output (H', W', num_filters).
        '''
        output = self._convolve(input_data) if not self.fast_convolution else self._convolve_fully_vectorized(input_data)
        return output
    
    def _convolve_fully_vectorized(self, input_data: np.ndarray) -> np.ndarray:
        '''
        Fully vectorized convolution operation with no loops.
        
        Args:
            input_data (np.ndarray): Input data (H, W, C).
        
        Returns:
            np.ndarray: Convolved output (H', W', num_filters).
        '''
        input_h, input_w, channels = input_data.shape
        kernel_h, kernel_w = self.filters.shape[1:3]
        output_h = input_h - kernel_h + 1
        output_w = input_w - kernel_w + 1
        
        # Create patches/windows for all positions in one go
        patches = np.zeros((output_h, output_w, kernel_h, kernel_w, channels))
        for h in range(kernel_h):
            for w in range(kernel_w):
                patches[:, :, h, w, :] = input_data[h:h+output_h, w:w+output_w, :]
        
        # Reshape patches to combine spatial dimensions for easy matrix multiplication
        patches_reshaped = patches.reshape(output_h * output_w, kernel_h * kernel_w * channels)
        
        # Reshape filters for matrix multiplication
        filters_reshaped = self.filters.reshape(self.num_filters, -1)
        
        # Compute convolution for all positions and all filters at once
        output = np.dot(patches_reshaped, filters_reshaped.T)
        
        # Reshape back to original spatial dimensions
        output = output.reshape(output_h, output_w, self.num_filters)
        
        return output
