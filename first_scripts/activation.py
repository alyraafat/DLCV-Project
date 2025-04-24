import numpy as np
from typing import Tuple, List

def relu(x: np.ndarray) -> np.ndarray:
    """Applies the ReLU activation function."""
    return np.maximum(0, x)