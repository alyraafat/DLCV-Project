from sklearn.cluster import KMeans
import numpy as np

def apply_kmeans(
    images: np.ndarray,
    num_clusters: int = 5,
    batch_size: int = 32,
    use_aug: bool = False,
    shuffle: bool = True
) -> np.ndarray:
    '''
    Applies KMeans clustering to the given images.

    Args:
        images (np.ndarray): Array of images.
        num_clusters (int): Number of clusters for KMeans.
        batch_size (int): Batch size for processing.
        use_aug (bool): Whether to apply data augmentation.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        np.ndarray: Cluster labels for each image.
    '''
    kmeans = KMeans(n_clusters=num_clusters, batch_size=batch_size, shuffle=shuffle)
    kmeans.fit(images)
    labels = kmeans.predict(images)  
    return labels
