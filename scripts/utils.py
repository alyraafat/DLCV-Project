import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

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
