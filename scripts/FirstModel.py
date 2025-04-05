from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import List, Tuple
from ConvLayer import ConvLayer
from PoolingLayer import PoolingLayer
from FlattenLayer import FlattenLayer
from tqdm import tqdm

class FirstModel:
    def __init__(self, 
                 conv_blocks: int = 3, 
                 num_filters: int = 5, 
                 kernel_size: Tuple[int,int] = (3, 3), 
                 pool_size: Tuple[int,int] = (2, 2),
                 pooling_type: str = "MAX"):
        '''
        Initializes a model with multiple convolution blocks followed by flattening and dimensionality reduction.
        '''
        self.conv_blocks = []
        for _ in range(conv_blocks):
            conv = ConvLayer(num_filters=num_filters, kernel_size=kernel_size)  
            pool = PoolingLayer(pooling_type=pooling_type, pool_size=pool_size)
            self.conv_blocks.append((conv, pool))
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
            for conv, pool in self.conv_blocks:
                x = conv.forward(x)
                x = pool.forward(x)
            flat = self.flatten.forward(x)
            print(f"Flattened shape: {flat.shape}")
            downsized = self._downsample(flat)
            feature_vectors.append(downsized)

        return np.array(feature_vectors)

    def _fit_kmeans(self, features: np.ndarray, num_clusters: int = 4):
        '''
        Runs KMeans on the extracted features.
        '''
        # scaler = StandardScaler()
        # features_scaled = scaler.fit_transform(features)
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        self.kmeans.fit(features)
    
    def fit(self, images: List[np.ndarray], num_clusters: int = 4):
        '''
        Extracts features from images and fits KMeans.
        '''
        features = self._extract_features(images)
        self._fit_kmeans(features, num_clusters)

    def predict(self, images: np.ndarray) -> np.ndarray:
        '''
        Predict cluster labels for given features.
        '''
        if self.kmeans is None:
            raise RuntimeError("You must call fit() before predicting.")
        features = self._extract_features(images)
        return self.kmeans.predict(features)
