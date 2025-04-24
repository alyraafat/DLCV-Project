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
                 kernel_size: Tuple[int,int] = (3, 3), 
                 pool_size: Tuple[int,int] = (2, 2),
                 pooling_type: str = "MAX", 
                 use_predefined_filters: bool = True):
        '''
        Initializes a model with multiple convolution blocks followed by flattening and dimensionality reduction.
        '''
        self.conv_blocks = []
        for _ in range(conv_blocks):
            if use_predefined_filters:
                conv = ConvLayer(num_filters=num_filters, kernel_size=kernel_size, filter_weights=self._get_predifined_filters())
            else:
                conv = ConvLayer(num_filters=num_filters, kernel_size=kernel_size)
            pool = PoolingLayer(pooling_type=pooling_type, pool_size=pool_size)
            relu = ReLU()
            self.conv_blocks.append((conv, pool, relu))
        
        self.flatten = FlattenLayer()
        self.kmeans = None

    def _downsample(self, flat_vector: np.ndarray, output_size: int = 128) -> np.ndarray:
        '''
        Downsamples a (1, D) vector to shape (1, output_size) using uniform sampling or zero-padding.
        '''
        original_size = flat_vector.shape[1]  
        indices = np.linspace(0, original_size - 1, output_size).astype(int)
        return flat_vector[:, indices]

      

    def _extract_features(self, images: List[np.ndarray]) -> np.ndarray:
        '''
        Pass all images through the convolutional blocks and return a matrix of features.
        
        Returns:
            np.ndarray: shape (num_images, 128)
        '''
        feature_vectors = []

        for img in tqdm(images):
            x = img.copy()
            for conv, pool, relu in self.conv_blocks:
                x = conv.forward(x)
                x = pool.forward(x)
                x = relu.forward(x)
            flat = self.flatten.forward(x)
            # print(f"Flattened shape: {flat.shape}")
            downsized = self._downsample(flat)
            # print(f"Downsized shape: {downsized.shape}")
            feature_vectors.append(downsized)

        return np.concatenate(feature_vectors, axis=0)

    def _fit_kmeans(self, features: np.ndarray, num_clusters: int = 4):
        '''
        Runs KMeans on the extracted features.
        '''
        # scaler = StandardScaler()
        # features_scaled = scaler.fit_transform(features)
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        self.kmeans.fit(features)
    
    def fit(self, images: List[np.ndarray], num_clusters: int = 5):
        '''
        Extracts features from images and fits KMeans.
        '''
        self.features = self._extract_features(images)
        print(f"Extracted features shape: {self.features.shape}")
        self._fit_kmeans(self.features, num_clusters)

    def predict(self, images: np.ndarray) -> np.ndarray:
        '''
        Predict cluster labels for given features.
        '''
        if self.kmeans is None:
            raise RuntimeError("You must call fit() before predicting.")
        features = self._extract_features(images)
        return self.kmeans.predict(features)
    
    def _get_predifined_filters(self):
        '''
        Returns a set of predefined filters for the convolutional layer.
        
        These filters are commonly used in image processing tasks.
        
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

        filter_a = np.stack([base_a]*3, axis=-1)
        filter_b = np.stack([base_b]*3, axis=-1)
        filter_c = np.stack([base_c]*3, axis=-1)
        filter_d = np.stack([base_d]*3, axis=-1)
        filter_e = np.stack([base_e]*3, axis=-1)


        filters = [filter_a, filter_b, filter_c, filter_d, filter_e]
        return filters
