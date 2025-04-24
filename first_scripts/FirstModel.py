from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import List, Tuple
from .ConvLayer import ConvLayer
from .PoolingLayer import PoolingLayer
from .FlattenLayer import FlattenLayer
from .ReLU import ReLU
from tqdm import tqdm

class FirstModel:
    def __init__(self, 
                 conv_blocks: int = 3, 
                 num_filters: int = 5, 
                 kernel_sizes: List[Tuple[int,int,int]] = [(3, 3, 3), (3, 3, 5), (3, 3, 5)], 
                 pool_size: Tuple[int,int] = (2, 2),
                 pooling_type: str = "MAX", 
                 use_predefined_filters: bool = True):
        '''
        Initializes a model with multiple convolution blocks followed by flattening and dimensionality reduction.
        '''
        assert conv_blocks == len(kernel_sizes), "conv_blocks must match the length of kernel_size"
        self.conv_blocks = []
        for i in range(conv_blocks):
            kernel = kernel_sizes[i]
            channel_dim = kernel_sizes[i][2]
            if use_predefined_filters:
                conv = ConvLayer(num_filters=num_filters, kernel_size=kernel, filter_weights=self._get_predifined_filters(channel_dim))
            else:
                conv = ConvLayer(num_filters=num_filters, kernel_size=kernel)
            pool = PoolingLayer(pooling_type=pooling_type, pool_size=pool_size)
            relu = ReLU()
            self.conv_blocks.append((conv, pool, relu))
        
        self.flatten = FlattenLayer()

    def _downsample(self, flat_vector: np.ndarray, output_size: int = 128) -> np.ndarray:
        '''
        Downsamples a (1, D) vector to shape (1, output_size) using uniform sampling or zero-padding.
        '''
        original_size = flat_vector.shape[1]  
        indices = np.linspace(0, original_size - 1, output_size).astype(int)
        return flat_vector[:, indices]

      

    def _extract_features(self, img: np.ndarray) -> np.ndarray:
        '''
        Pass all images through the convolutional blocks and return a matrix of features.
        
        Returns:
            np.ndarray: shape (num_images, 128)
        '''

        x = img.copy()
        for conv, pool, relu in self.conv_blocks:
            x = conv.forward(x)
            x = pool.forward(x)
            x = relu.forward(x)
        flat = self.flatten.forward(x)
        downsized = self._downsample(flat)

        return downsized


    def forward(self, images: List[np.ndarray]) -> np.ndarray:
        '''
        Extracts features from images and fits KMeans.
        '''
        features = self._extract_features(images)
        return features

    
    def _get_predifined_filters(self, channel_dim: int):
        '''
        Returns a set of predefined filters for the convolutional layer.
        
        These filters are commonly used in image processing tasks.
        
        Args:
            channel_dim (int): Number of channels in the input image.

        Returns:
            List[np.ndarray]: List of predefined filters.
        '''
        base_a = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ])

        base_b = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])

        base_c = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])

        base_d = np.array([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ])

        base_e = np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ])

        filter_a = np.stack([base_a]*channel_dim, axis=-1)
        filter_b = np.stack([base_b]*channel_dim, axis=-1)
        filter_c = np.stack([base_c]*channel_dim, axis=-1)
        filter_d = np.stack([base_d]*channel_dim, axis=-1)
        filter_e = np.stack([base_e]*channel_dim, axis=-1)


        filters = [filter_a, filter_b, filter_c, filter_d, filter_e]
        return filters
