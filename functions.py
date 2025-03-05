import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z)) 

def unit_step(z: np.ndarray, threshold = 0.5) -> np.ndarray:
    return (z > threshold).astype(int)

def ReLU(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)