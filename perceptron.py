import numpy as np
from functions import unit_step

class PerceptronClassifier():
    def __init__(self, lr: float = 1e-3, iterations: int = 100):
        self.W = None
        self.b = None
        self.lr = lr
        self.iterations = iterations
 
    def train(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

        self.W = np.random.rand(x.shape[1])  
        self.b = 0  

        for _ in range(self.iterations):
            for idx, x_i in enumerate(x):
                z = np.dot(x_i, self.W) + self.b 
                y_pred = unit_step(z)

                dw = (y[idx] - y_pred) * x_i
                db = y[idx] - y_pred

                self.W += self.lr * dw  
                self.b += self.lr * db 

    def predict(self, instance: np.ndarray):
         z = np.dot(instance, self.W) + self.b
         return unit_step(z)

                