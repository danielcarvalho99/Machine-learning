import numpy as np
from math import sqrt
from statistics import mode

class KNNClassifier():
    def __init__(self, neighbors: int):
        self.neighbors = neighbors

    def train(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    def predict(self, instance: np.ndarray) -> int:
        distances = np.sqrt(np.sum((self.x - instance) ** 2, axis=1))
        idx_sorted = np.argsort(distances)[:self.neighbors]
        knn_labels = self.y[idx_sorted]
        return mode(knn_labels)

        
