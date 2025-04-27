import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle as pkl


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



# def plot_feature_scatter(
#     features: np.ndarray,
#     labels: np.ndarray,
#     n_components: int = 2,
# ):
#     """
#     Plot PCA-reduced features in 2D or 3D based on n_components.

#     Args:
#         features (np.ndarray): Feature matrix of shape (N, D).
#         labels (np.ndarray): Cluster labels of shape (N,).
#         n_components (int): 2 for 2D plot, 3 for 3D plot.
#     """
#     proj = PCA(n_components=n_components).fit_transform(features)

#     figsize = (20,20)

#     fig = plt.figure(figsize=figsize)

#     if n_components == 2:
#         ax = fig.add_subplot(111)
#         scatter = ax.scatter(proj[:, 0], proj[:, 1], c=labels, cmap='tab10', s=20)
#         handles, legend_labels = scatter.legend_elements()
#         ax.legend(handles, legend_labels, title="Cluster")
#         ax.set_xlabel("PC1")
#         ax.set_ylabel("PC2")
#         ax.set_title("2D PCA Projection of Cluster Assignments")

#     elif n_components == 3:
#         ax = fig.add_subplot(111, projection='3d')
#         scatter = ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], c=labels, cmap='tab10', s=20)
#         handles, legend_labels = scatter.legend_elements()
#         ax.legend(handles, legend_labels, title="Cluster")
#         ax.set_xlabel("PC1")
#         ax.set_ylabel("PC2")
#         ax.set_zlabel("PC3")
#         ax.set_title("3D PCA Projection of Cluster Assignments")

#     else:
#         raise ValueError("n_components must be 2 or 3 for plotting")

#     plt.show()

    

def save_model(model, filename: str) -> None:
    """
    Save the model to a file using pickle.

    Args:
        model: The model to save.
        filename (str): The name of the file to save the model to.
    """
    with open(filename, 'wb') as f:
        pkl.dump(model, f)


def load_model(filename: str):
    """
    Load the model from a file using pickle.

    Args:
        filename (str): The name of the file to load the model from.

    Returns:
        The loaded model.
    """
    with open(filename, 'rb') as f:
        model = pkl.load(f)
    return model


def plot_predictions(
    images: List,
    true_labels: List[int],
    pred_labels: List[int],
    index_to_label: Dict[int, str],
    num_images: int = 16,
    nrow: int = 4,
    mean: List[float] = None,
    std: List[float] = None,
):
    """
    Plot a grid of images with predicted vs true labels, coloring the title green if correct, red if wrong.

    Args:
        images (List[np.ndarray]): List of images (H, W, C).
        true_labels (List[int]): List of true label indices.
        pred_labels (List[int]): List of predicted label indices.
        index_to_label (dict): Mapping from index to label name.
        num_images (int): How many images to plot.
        nrow (int): Number of images per row.
        mean (List[float], optional): Mean values for normalization.
        std (List[float], optional): Standard deviation values for normalization.
    """
    if mean is not None and std is not None:
        mean = np.array(mean)[None,None,:]
        std  = np.array(std)[None,None,:]
    num = min(len(images), num_images)
    ncol = (num + nrow - 1) // nrow
    plt.figure(figsize=(nrow * 3, ncol * 3))
    for i in range(num):
        ax = plt.subplot(ncol, nrow, i + 1)
        img = images[i]           # H×W×C, but still normalized
        if mean is not None and std is not None:
            img = img * std + mean    # un-do Normalize
            img = np.clip(img, 0, 1)  # ensure within [0..1]
        plt.imshow(img)
        correct = (true_labels[i] == pred_labels[i])
        color = 'green' if correct else 'red'
        title_text = f"P: {index_to_label[pred_labels[i]]} / T: {index_to_label[true_labels[i]]}"
        ax.set_title(title_text, color=color, fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.show()