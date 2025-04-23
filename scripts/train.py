import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import CrossEntropyLoss
from torch import nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import List, Tuple, Dict
from tqdm import tqdm

def training(
        model: nn.Module, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        criterion: CrossEntropyLoss, 
        optimizer: optim, 
        num_epochs: int) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Train the model using the provided data loaders, loss function, and optimizer.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        criterion (CrossEntropyLoss): Loss function to be used.
        optimizer (optim): Optimizer to be used for training.
        num_epochs (int): Number of epochs to train the model.

    Returns:
        Tuple[List[float], List[float], List[float], List[float]]: 
        - train_losses: List of training losses for each epoch.
        - val_losses: List of validation losses for each epoch.
        - train_accuracies: List of training accuracies for each epoch.
        - val_accuracies: List of validation accuracies for each epoch.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, desc=f"Train {epoch}/{num_epochs}", leave=False)
        for images, labels in train_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            train_bar.set_postfix(
                loss=f"{train_loss/total:.4f}", 
                acc=f"{correct/total:.4f}"
            )

        train_accuracy = correct / total
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_accuracy)
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f"train_loss: {train_loss/total:.4f}, train_acc: {train_accuracy:.4f}")

        # train_metrics = evaluate(model, train_loader, criterion, device, prefix='train_')

        val_metrics = evaluate(model, val_loader, criterion, device, prefix='val_')
        val_losses.append(val_metrics['loss'])
        val_accuracies.append(val_metrics["accuracy"])

        print('-' * 50)

    return train_losses, val_losses, train_accuracies, val_accuracies



def evaluate(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device = None,
    average: str = 'macro',
    prefix: str = 'val_',
) -> Dict[str, float]:
    """
    Evaluate the model on `data_loader` and return loss, accuracy, precision, recall and F1.

    Args:
        model (nn.Module): Trained model.
        data_loader (DataLoader): DataLoader for the evaluation set.
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).
        device (torch.device, optional): If None, inferred from model.parameters().
        average (str): Averaging method for precision/recall/f1 ('macro', 'weighted', etc.).
        prefix (str): Prefix for the metrics keys.

    Returns:
        Dict[str, float]: {
            'loss': float,
            'accuracy': float,
            'precision': float,
            'recall': float,
            'f1': float
        }
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        bar = tqdm(data_loader, leave=False)
        for images, labels in bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())


    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average=average, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=average, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=average, zero_division=0)
    
    print(f'{prefix}Loss: {avg_loss:.4f}, {prefix}Accuracy: {accuracy:.4f}, {prefix}Precision: {precision:.4f}, {prefix}Recall: {recall:.4f}, {prefix}F1: {f1:.4f}')

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
