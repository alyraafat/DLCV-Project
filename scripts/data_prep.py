import cv2 
import os 
import numpy as np
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms


def data_reader(data_path: str) -> Tuple[List[np.ndarray], List[str]]:
    ''''
    Reads images from the given directory and returns a list of images, and get the labels from the directory names.

    Args:
        data_path (str): Path to the directory containing images.
    Returns:
        List[np.ndarray]: List of images read from the directory.
    '''
    images, labels = [], []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                img_path = os.path.join(root, file)
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
                label = os.path.basename(root)
                labels.append(label)
    return images, labels
    


def data_preperator(data: Tuple[List[np.ndarray], List[str]], image_size: Tuple[int, int] = (512, 512)) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Dict[str, int]]:
    '''
    Prepares the data for training by resizing images and converting labels to one-hot encoding.

    Args:
        data (Tuple[List[np.ndarray], List[str]]): Tuple containing a list of images and their corresponding labels.
        image_size (Tuple[int, int]): Desired size for the images after resizing.

    Returns:
        Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Dict[str, int]]: 
        Tuple containing training data, validation data, test data, and a mapping of labels to indices.
    '''
    images, labels = data
    
    unique_labels = sorted(list(set(labels)))
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    labels_encoded = np.array([label_to_index[label] for label in labels])
    
    idx = np.arange(len(images))
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        idx, labels_encoded,
        test_size=0.30,
        random_state=42,
        stratify=labels_encoded
    )
    val_idx, test_idx, y_val, y_test = train_test_split(
        temp_idx, y_temp,
        train_size=2/3,
        random_state=42,
        stratify=y_temp
    )

    def build_subset(idxs: List[int]) -> np.ndarray:
        '''
        Builds a subset of images based on the provided indices.

        Args:
            idxs (List[int]): List of indices to select images from.
        
        Returns:
            np.ndarray: Array of images resized to the specified image size.
        '''
        
        X = np.array([cv2.resize(images[i], image_size) / 255.0 for i in idxs], dtype=np.float32)
        return X
    
    X_train = build_subset(train_idx)
    X_val = build_subset(val_idx)
    X_test = build_subset(test_idx)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), label_to_index
    
    
   
def data_augmentor(data: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Applies data augmentation techniques to the training data.

    Args:
        data (Tuple[np.ndarray, np.ndarray]): Tuple containing the training images and their corresponding labels.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Augmented training images and their corresponding labels.
    '''
    images, labels = data
    augmented_images = []

    for image in images:
        # Random brightness adjustment
        if np.random.rand() > 0.5:
            brightness_factor = np.random.uniform(0.9,1.1)
            image = np.clip(image * brightness_factor, 0.0, 1.0)

        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)

        # Random vertical flip
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 0)
        
        # Random rotation
        if np.random.rand() > 0.5:
            angle = np.random.randint(-15, 15)
            M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1)
            image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        
        augmented_images.append(image)

    return np.array(augmented_images), labels


def convert_to_dataloader(data: Tuple[np.ndarray, np.ndarray], batch_size: int=32, transform: transforms.Compose=None, use_aug: bool=False) -> DataLoader:
    '''
    Converts the data to a format suitable for PyTorch.

    Args:
        data (Tuple[np.ndarray, np.ndarray]): Tuple containing the images and their corresponding labels.
        batch_size (int): Size of the batches for the DataLoader.
        transform (transforms.Compose): Transformations to be applied to the images.
        use_aug (bool): Whether to apply data augmentation.

    Returns:
        DataLoader: PyTorch DataLoader containing the images and labels.
    '''
    images, labels = data
    
    if use_aug:
        images, labels = data_augmentor(data)

    if transform:
        images = [transform(image) for image in images]
        data_tensor = torch.stack(images)
    else:
        data_tensor = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)

    labels_tensor = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(data_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader