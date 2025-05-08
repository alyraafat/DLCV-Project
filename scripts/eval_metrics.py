import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from typing import List, Tuple, Dict

def plot_acc_loss_curves_train_vs_val(train_losses: List[float], val_losses: List[float], train_accuracies: List[float], val_accuracies: List[float]):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_conf_mtrx(y_true: List[int], y_pred: List[int], classes: List[str], title: str = "Confusion Matrix") -> None:
    """
    Plot the confusion matrix using seaborn heatmap.

    Args:
        y_true (List[int]): True labels.
        y_pred (List[int]): Predicted labels.   
        classes (List[str]): List of class names.
        title (str): Title for the plot.
    """
    cm = confusion_matrix_from_scratch(y_true, y_pred, classes=classes)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    print(classification_report(y_true, y_pred, target_names=classes))

def confusion_matrix_from_scratch(
    y_true: List[int],
    y_pred: List[int],
    classes: List[str],
) -> Tuple[np.ndarray]:
    """
    Compute the confusion matrix.

    Args:
        y_true:   list of true labels
        y_pred:   list of predicted labels
        classes:   list of class names
    Returns:
        cm:       numpy array of shape (C, C) where C = len(labels);
                  cm[i, j] = count of samples with true label labels[i] predicted as labels[j]
    """
    C = len(classes)
    cm = np.zeros((C, C), dtype=int)

    for i, j in zip(y_true, y_pred):
        cm[i, j] += 1

    return cm