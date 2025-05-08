import numpy as np
import pandas as pd
from typing import Union, List

class KMeans:
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, init_centroids: List[float]=None):
        assert n_clusters > 0, "Number of clusters must be greater than 0"
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = init_centroids if init_centroids is not None else None 

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> None:
        """
        Fit the KMeans model
        
        Params:
          - X: Input features (numpy array or pandas DataFrame)
        """
        self.centroids = self._random_init(X) if self.centroids is None else self.centroids
        for _ in range(self.max_iter):
            prev_centroids = self.centroids.copy() # Copy the current centroids
            labels = self._assign_clusters(X) # Assign the input features to the closest cluster
            self.centroids = self._update_centroids(X, labels) # Update the centroids of the clusters
            if np.allclose(prev_centroids, self.centroids, atol=self.tol):  # Check for convergence
                break


    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        '''
        Predict the cluster labels for the input features
        
        Params:
          - X: Input features (numpy array or pandas DataFrame)
        '''
        labels = self._assign_clusters(X)
        return labels 
        
    def predict_soft(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        '''
        Predict the soft cluster labels for the input features
        
        Params:
          - X: Input features (numpy array or pandas DataFrame)
        '''
        labels = self._assign_soft_clusters(X)
        return labels
    
    def _random_init(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize the centroids randomly
        
        Params:
          - X: Input features (numpy array)
        """
        centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)] # Randomly select k points as centroids
        return centroids


    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """
        Assign the input features to the closest cluster
        
        Params:
          - X: Input features (numpy array)
        """
        new_labels = np.zeros(X.shape[0], dtype=float) # Initialize the labels
        for i in range(X.shape[0]): 
            new_labels[i] = np.argmin(np.linalg.norm(X[i] - self.centroids, axis=1)) # Assign the input feature to the closest cluster, based on the Euclidean distance
        return new_labels
        

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Update the centroids of the clusters
        
        Params:
          - X: Input features (numpy array)
        """
        new_centroids = np.zeros((self.n_clusters, X.shape[-1])) # Initialize the new centroids
        for i in range(self.n_clusters):
            new_centroids[i] = np.mean(X[labels == i], axis=0) # Update the centroid of the cluster as the mean of the input features assigned to it
        return new_centroids

    def _assign_soft_clusters(self, X: np.ndarray) -> np.ndarray:
        """
        Assign the input features to the closest cluster using soft clustering

        Params:
          - X: Input features (numpy array)
        """
        new_labels = np.zeros((X.shape[0], self.n_clusters), dtype=float) 
        for i in range(X.shape[0]): 
            new_labels[i] = np.exp(-np.linalg.norm(X[i] - self.centroids, axis=1)) # Assign the input feature to the closest cluster, based on the Euclidean distance
        return new_labels / np.sum(new_labels, axis=1, keepdims=True) # Normalize the labels to sum to 1
      


        