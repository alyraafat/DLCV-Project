import cv2 
import os 
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import train_test_split

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
    


def data_preperator(data: Tuple[List[np.ndarray], List[str]], image_size: Tuple[int, int] = (512, 512)) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    '''
    Prepares the data for training by resizing images and converting labels to one-hot encoding.

    Args:
        data (Tuple[List[np.ndarray], List[str]]): Tuple containing a list of images and their corresponding labels.
        image_size (Tuple[int, int]): Desired size for the images after resizing.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing the preprocessed images and their corresponding labels.
    '''
    images, labels = data
    images_resized_normalized = [cv2.resize(image, image_size) / 255.0 for image in images]
    images_array = np.array(images_resized_normalized)
    
    unique_labels = sorted(list(set(labels)))
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    labels_encoded = np.array([label_to_index[label] for label in labels])
    
    X_train, X_temp, y_train, y_temp = train_test_split(images_array, labels_encoded, test_size=0.3, random_state=42, stratify=labels_encoded)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=2/3, random_state=42, stratify=y_temp)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    
   
     