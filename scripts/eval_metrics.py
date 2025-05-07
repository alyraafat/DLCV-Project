import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from typing import List, Tuple

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


def plot_conf_mtrx(y_true: List[int], y_pred: List[int], classes: List[str], title: str = "Confusion Matrix"):
    """
    Plot the confusion matrix using seaborn heatmap.

    Args:
        y_true (List[int]): True labels.
        y_pred (List[int]): Predicted labels.
        classes (List[str]): List of class names.
        title (str): Title for the plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    print(classification_report(y_true, y_pred, target_names=classes))