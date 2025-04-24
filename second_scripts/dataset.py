import os
import random
from glob import glob
from typing import List, Tuple, Dict

import numpy as np
import cv2
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageClassificationDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[str,int]],
        transform: transforms.Compose=None,
        augment: bool = False,
        image_size: Tuple[int,int] = (512, 512)
    ):
        """
        samples: list of (image_path, label_index)
        transform: optional torchvision transforms to apply to the image
        augment: whether to apply random on‐the‐fly augmentations
        image_size: target size for resizing images 
        """
        self.samples   = samples
        self.transform = transform
        self.augment   = augment
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, lbl = self.samples[idx]

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = cv2.resize(img, self.image_size)  
        img /= 255.0  # normalize to [0, 1]

        if self.augment:
            # brightness
            if random.random() > 0.5:
                factor = random.uniform(0.9, 1.1)
                img = np.clip(img * factor, 0.0, 1.0)
            # flips
            if random.random() > 0.5:
                img = cv2.flip(img, 1)  # horizontal flip
            if random.random() > 0.5:
                img = cv2.flip(img, 0)  # vertical flip
            # rotation
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # HWC to CHW
        if self.transform:
            img = self.transform(img)

        label = torch.tensor(lbl, dtype=torch.long)
        return img, label

def make_datasets(
    data_path: str,
    image_size: Tuple[int,int] = (512,512),
    test_size: float = 0.30,
    val_size: float = 1/3,    # fraction *of the remaining* after test split
    random_state: int = 42,
    augment: bool = False
) -> Tuple[ImageClassificationDataset, ImageClassificationDataset, ImageClassificationDataset, Dict[str,int]]:
    # 1) gather all files and labels
    samples: List[Tuple[str,str]] = []
    for class_name in sorted(os.listdir(data_path)):
        class_dir = os.path.join(data_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        for ext in ("*.jpg", "*.png"):
            for path in glob(os.path.join(class_dir, ext)):
                samples.append((path, class_name))

    # 2) map label names → integers
    classes = sorted({lab for _, lab in samples})
    label_to_index = {lab:i for i, lab in enumerate(classes)}
    samples_idx = [(p, label_to_index[lab]) for p, lab in samples]

    # 3) stratified split on *indices*
    paths, labels = zip(*samples_idx)
    idx = np.arange(len(paths))
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        idx, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    val_idx, test_idx, y_val, y_test = train_test_split(
        temp_idx, y_temp,
        train_size= 1 - val_size,
        random_state=random_state,
        stratify=y_temp
    )

    # 4) build sample lists per split
    train_samples = [samples_idx[i] for i in train_idx]
    val_samples   = [samples_idx[i] for i in val_idx]
    test_samples  = [samples_idx[i] for i in test_idx]

    # 6) create Dataset objects
    train_ds = ImageClassificationDataset(train_samples, augment=augment, image_size=image_size)
    val_ds   = ImageClassificationDataset(val_samples, augment=False, image_size=image_size)
    test_ds  = ImageClassificationDataset(test_samples, augment=False, image_size=image_size)

    return train_ds, val_ds, test_ds, label_to_index


def make_dataloader(
    dataset: ImageClassificationDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    Create a DataLoader for the given dataset.

    Args:
        dataset (ImageClassificationDataset): The dataset to load data from.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data at every epoch.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        DataLoader: DataLoader for the given dataset.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# USAGE EXAMPLE:
#
# train_ds, val_ds, test_ds, lbl2idx = make_datasets("/path/to/data")
# train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
# val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)
# test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)
#
# # now `train_loader` yields (images, labels) with on‐the‐fly augmentation,
# # correct resize, normalization, and channel‐first format.
