import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def draw_barchart_labels(labels: np.ndarray, label_to_index: Dict[str, int]) -> None:
    """
    Draw a bar chart of the number of samples per class in the labels array.

    Args:
        labels (np.ndarray): Array of numeric labels.
        label_to_index (Dict[str, int]): Dictionary mapping class names to indices.
    """
    index_to_label = {v: k for k, v in label_to_index.items()}

    unique_indices, counts = np.unique(labels, return_counts=True)

    class_names = [index_to_label[i] for i in unique_indices]

    plt.figure(figsize=(8, 6))
    plt.bar(class_names, counts, color='skyblue')
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.title("Number of Samples per Class in train_labels")
    plt.show()



def plot_loss_acc(
        train_losses: List[float], 
        val_losses: List[float], 
        train_accuracies: List[float], 
        val_accuracies: List[float]):
    """
    Plot training vs validation loss and accuracy over epochs.

    Args:
        train_losses (List[float]): Training loss per epoch.
        val_losses (List[float]): Validation loss per epoch.
        train_accuracies (List[float]): Training accuracy per epoch.
        val_accuracies (List[float]): Validation accuracy per epoch.
    """
    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    plt.figure()
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs, val_losses,   marker='o', label='Val Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Accuracy plot
    plt.figure()
    plt.plot(epochs, train_accuracies, marker='o', label='Train Accuracy')
    plt.plot(epochs, val_accuracies,   marker='o', label='Val Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_feature_scatter(
    features: np.ndarray,
    labels: np.ndarray,
    n_components: int = 2,
):
    """
    Plot PCA-reduced features in 2D or 3D based on n_components.

    Args:
        features (np.ndarray): Feature matrix of shape (N, D).
        labels (np.ndarray): Cluster labels of shape (N,).
        n_components (int): 2 for 2D plot, 3 for 3D plot.
    """
    proj = PCA(n_components=n_components).fit_transform(features)

    figsize = (20,20)

    fig = plt.figure(figsize=figsize)

    if n_components == 2:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(proj[:, 0], proj[:, 1], c=labels, cmap='tab10', s=20)
        handles, legend_labels = scatter.legend_elements()
        ax.legend(handles, legend_labels, title="Cluster")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("2D PCA Projection of Cluster Assignments")

    elif n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], c=labels, cmap='tab10', s=20)
        handles, legend_labels = scatter.legend_elements()
        ax.legend(handles, legend_labels, title="Cluster")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title("3D PCA Projection of Cluster Assignments")

    else:
        raise ValueError("n_components must be 2 or 3 for plotting")

    plt.show()

    

