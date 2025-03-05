import numpy as np

def sigmoid(z: np.ndarray):
    return 1 / (1 + np.exp(-z)) 

def unit_step(z: np.ndarray, threshold = 0.5):
    return (z > threshold).astype(int)