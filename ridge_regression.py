import numpy as np
from utils import ridge_loss

class RidgeRegression():
    def __init__(self, lr: float=1e-3, iterations: int=1000, lambda_val: float=1e-3):
        self.lr = lr
        self.iterations = iterations
        self.W = None
        self.b = None
        self.lambda_val = lambda_val
        self.loss = None

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.y = y.reshape(-1,1)
        m, n = self.x.shape

        self.W = np.random.rand(n, 1)
        self.b = np.zeros((1, 1))

        for i in range(self.iterations):
            for _, w_i in enumerate(self.x):
                y_pred = np.dot(self.x, self.W) + self.b
                self.loss = ridge_loss(self.y, y_pred, w_i, self.lambda_val)
                
                dw = -2/m * np.dot(self.x.T ,(self.y - y_pred)) + 2 * self.lambda_val * w_i
                db = -2/m * (np.sum(y - y_pred))
                
                self.W -= self.lr * dw
                self.b -= self.lr * db
    
    def predict(self, instance: np.ndarray) -> int:
        y_pred = np.dot(instance, self.W) + self.b
        return y_pred