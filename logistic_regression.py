import numpy as np
from utils import compute_loss
from functions import sigmoid

class LogisticRegressionClassifier():
    def __init__(self, lr: float = 1e-3, iterations: int = 100):
        self.W = None
        self.b = None
        self.lr = lr
        self.iterations = iterations
 
    def train(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y.reshape(-1, 1)  

        self.W = np.random.rand(x.shape[1], 1)  
        self.b = 0  

        for _ in range(self.iterations):
            m = x.shape[0]

            z = np.dot(self.x, self.W) + self.b 
            y_pred = sigmoid(z)

            dw = (1/m) * np.dot(x.T, (y_pred - self.y))  
            db = (1/m) * np.sum(y_pred - self.y) 

            self.W -= self.lr * dw  
            self.b -= self.lr * db  

            compute_loss(self.y, y_pred)
    
    def predict(self, instance: np.ndarray):
         z = np.dot(instance.reshape(1, -1), self.W) + self.b
         y_pred = 1 / (1 + np.exp(-z))
         return int(y_pred > 0.5)