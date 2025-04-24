import cv2 
import os 
import numpy as np
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
import random

def data_reader(data_path: str) -> Tuple[List[np.ndarray], List[str]]:
    ''''
    Reads images from the given directory and returns a list of images, and get the labels from the directory names.

    Args:
        data_path (str): Path to the directory containing images.
    Returns:
        Tuple[List[np.ndarray], List[str]]: Tuple containing a list of img_paths and their corresponding labels.
    '''
    images, labels = [], []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                img_path = os.path.join(root, file)
                images.append(img_path)
                label = os.path.basename(root)
                labels.append(label)
    return images, labels


def read_img(file_path: str):
    '''
    Reads an image from the given file path and converts it to RGB format.

    Args:
        file_path (str): Path to the image file.

    Returns:
        np.ndarray: Image in RGB format.
    '''
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))
    image = image.astype(np.float32) / 255.0
    return image
    

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
    paths, labels = data
    
    unique_labels = sorted(list(set(labels)))
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    labels_encoded = np.array([label_to_index[label] for label in labels])
    
    # idx = np.arange(len(images))
    train_path, temp_path, y_train, y_temp = train_test_split(
        paths, labels_encoded,
        test_size=0.30,
        random_state=42,
        stratify=labels_encoded
    )
    val_path, test_path, y_val, y_test = train_test_split(
        temp_path, y_temp,
        train_size=2/3,
        random_state=42,
        stratify=y_temp
    )

    # def build_subset(idxs: List[int]) -> np.ndarray:
    #     '''
    #     Builds a subset of images based on the provided indices.

    #     Args:
    #         idxs (List[int]): List of indices to select images from.
        
    #     Returns:
    #         np.ndarray: Array of images resized to the specified image size.
    #     '''
        
    #     X = np.array([cv2.resize(images[i], image_size) / 255.0 for i in idxs], dtype=np.float32)
    #     return X
    
    # X_train = build_subset(train_idx)
    # X_val = build_subset(val_idx)
    # X_test = build_subset(test_idx)

    train_path = np.array(train_path)
    val_path = np.array(val_path)
    test_path = np.array(test_path)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    return (train_path, y_train), (val_path, y_val), (test_path, y_test), label_to_index
    
    
   
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


def convert_to_dataloader(data: Tuple[np.ndarray, np.ndarray], batch_size: int=32, transform: transforms.Compose=None, use_aug: bool=False, shuffle: bool=False) -> DataLoader:
    '''
    Converts the data to a format suitable for PyTorch.

    Args:
        data (Tuple[np.ndarray, np.ndarray]): Tuple containing the images and their corresponding labels.
        batch_size (int): Size of the batches for the DataLoader.
        transform (transforms.Compose): Transformations to be applied to the images.
        use_aug (bool): Whether to apply data augmentation.
        shuffle (bool): Whether to shuffle the data.

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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader



class AugmentedImageDataset(Dataset):
    def __init__(
        self,
        images: List[np.ndarray],
        labels: List[int],
        transform=None,
        augment: bool = False
    ):
        """
        images: list of H×W×C numpy arrays (normalized to [0,1])
        labels: list of ints
        transform: optional callable to convert numpy array to tensor and normalize
        augment: whether to apply random augmentations
        """
        self.images = images
        self.labels = labels
        self.transform = transform

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        
        self.augment = augment

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        image = self.images[idx].copy()
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long)

        if self.augment:
            # Random brightness
            if random.random() > 0.5:
                factor = random.uniform(0.9, 1.1)
                image = np.clip(image * factor, 0.0, 1.0)

            # Horizontal flip
            if random.random() > 0.5:
                image = cv2.flip(image, 1)

            # Vertical flip
            if random.random() > 0.5:
                image = cv2.flip(image, 0)

            # Rotation
            if random.random() > 0.5:
                angle = random.randint(-15, 15)
                h, w = image.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        if self.transform:
            image = self.transform(image)

        return image, label

def convert_to_dataloader_optimized(data, batch_size=32, transform=None, use_aug=False, shuffle=True):
    images, labels = data
    dataset = AugmentedImageDataset(images, labels, transform=transform, augment=use_aug)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

