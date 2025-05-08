import torch
from torch.utils.data import DataLoader, Dataset
from torch import optim
from torch.nn import CrossEntropyLoss
from torch import nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import List, Tuple, Dict, Iterator, Sequence
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import Subset, DataLoader
import torch.nn.functional as F

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

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        train_metrics = evaluate(model, train_loader, criterion, device, prefix='train_')
        train_accuracy = correct / total
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_accuracy)        
        # print(f"train_loss: {train_loss/total:.4f}, train_acc: {train_accuracy:.4f}")

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
    should_print: bool = True,
) -> Dict[str, float]:
    """
    Evaluate the model and return loss, accuracy, precision, recall, f1, and roc_auc.
    """

    if device is None:
        device = next(model.parameters()).device
    model.eval()

    all_preds = []
    all_labels = []
    all_scores = []      
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(data_loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # 1) hard predictions:
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            # 2) softmax scores for ROC-AUC:
            probs = F.softmax(logits, dim=1)
            all_scores.append(probs.cpu())

    all_preds  = torch.cat(all_preds).numpy()       # shape (N,)
    all_labels = torch.cat(all_labels).numpy()      # shape (N,)
    all_scores = torch.cat(all_scores).numpy()      # shape (N, C)

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average=average, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=average, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=average, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_scores, multi_class='ovr', average=average)

    if should_print:
        print(
            f"{prefix}Loss: {avg_loss:.4f}  "
            f"{prefix}Acc: {accuracy:.4f}  "
            f"{prefix}Prec: {precision:.4f}  "
            f"{prefix}Rec: {recall:.4f}  "
            f"{prefix}F1: {f1:.4f}  "
            f"{prefix}ROC-AUC: {roc_auc:.4f}"
        )

    return {
        'loss':      avg_loss,
        'accuracy':  accuracy,
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
        'roc_auc':   roc_auc
    }


def predict(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device = None,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Run model on `dataloader` and collect images, true labels, and predicted labels.

    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): DataLoader for the dataset to be evaluated.
        device (torch.device, optional): Device to run on.

    Returns:
        Tuple[List[np.ndarray], List[int], List[int]]:
            - images: list of numpy arrays (H, W, C)
            - true_labels: list of true label indices
            - pred_labels: list of predicted label indices
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    images_list, true_list, pred_list = [], [], []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)

            imgs_cpu = imgs.cpu()
            labels_cpu = labels.cpu()
            preds_cpu = preds.cpu()

            for img_tensor, true_lbl, pred_lbl in zip(imgs_cpu, labels_cpu, preds_cpu):
                # convert C×H×W → H×W×C for plotting
                img_np = img_tensor.permute(1, 2, 0).numpy()
                images_list.append(img_np)
                true_list.append(true_lbl.item())
                pred_list.append(pred_lbl.item())

    return images_list, true_list, pred_list



def cross_validate(
    train_ds: Dataset,
    model_fn: callable,
    k: int = 4,
    num_epochs: int = 10,
    batch_size: int = 32,
    num_workers: int = 4,
    lr: float = 1e-3,
    random_state: int = 42,
) -> Tuple[List[List[float]], List[List[float]], List[List[float]], List[List[float]], nn.Module]:
    """
    Perform K-fold (or Stratified K-fold) cross-validation on `train_ds`.
    
    Args:
        train_ds        : a torch.utils.data.Dataset containing all your training samples.
        model_fn        : a callable with zero args that returns a fresh nn.Module.
        k               : number of folds (>=4).
        num_epochs      : epochs per fold.
        batch_size      : batch size.
        num_workers     : dataloader num_workers.
        lr              : learning rate for Adam.
        random_state    : seed for shuffling.
    
    Returns:
        Tuple[List[List[float]], List[List[float]], List[List[float]], List[List[float]]]:
            - fold_train_losses: list of training losses for each fold.
            - fold_val_losses: list of validation losses for each fold.
            - fold_train_accuracies: list of training accuracies for each fold.
            - fold_val_accuracies: list of validation accuracies for each fold.
            - best_model: the best model from all folds.
    """

    splits = kfold_indices(len(train_ds), k, random_state=random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_train_losses = []
    fold_val_losses = []
    fold_train_accuracies = []
    fold_val_accuracies = []
    fold_last_val_accuracies = 0
    best_acc = 0.0
    best_model = None

    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        print(f"\n--- Fold {fold}/{k} ---")
        train_sub   = Subset(train_ds, train_idx)
        val_sub     = Subset(train_ds, val_idx)
        train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
        val_loader   = DataLoader(val_sub,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

        model     = model_fn().to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        train_losses, val_losses, train_accuracies, val_accuracies = training(
            model, train_loader, val_loader, criterion, optimizer, num_epochs
        )

        fold_train_losses.append(train_losses)
        fold_val_losses.append(val_losses)
        fold_train_accuracies.append(train_accuracies)
        fold_val_accuracies.append(val_accuracies)

        fold_acc = val_accuracies[-1]
        print(f"Fold {fold} val accuracy: {fold_acc:.4f}")
        fold_last_val_accuracies += fold_acc
        if fold_acc > best_acc:
            best_acc = fold_acc
            best_model = model

    print(f"\nBest fold validation accuracy: {best_acc:.4f}")
    avg_acc = fold_last_val_accuracies / k
    print(f"\nAverage validation accuracy over {k} folds: {avg_acc:.4f}")
    return fold_train_losses, fold_val_losses, fold_train_accuracies, fold_val_accuracies, best_model

def kfold_indices(
    n_samples: int, 
    k: int, 
    random_state: int = 42
) -> Iterator[Tuple[List[int], List[int]]]:
    """
    Yield train/val index splits for standard K-fold.
    """
    rng = np.random.RandomState(random_state)
    indices = np.arange(n_samples)

    base_size = n_samples // k
    sizes = [base_size + (1 if i < (n_samples % k) else 0) for i in range(k)]
    print(f"Fold sizes: {sizes}")

    folds = []
    start = 0
    for size in sizes:
        folds.append(indices[start:start+size])
        start += size

    for i in range(k):
        val_idx = folds[k-i-1].tolist()
        train_idx = np.hstack([folds[j] for j in range(k) if j != k-i-1]).tolist()
        yield train_idx, val_idx