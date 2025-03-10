import numpy as np
from utils import elastic_net_loss

class ElasticNetRegression():
    def __init__(self, lr: float=1e-3, iterations: int=1000, lambda_1: float=1e-3, lambda_2: float=1e-3):
        self.lr = lr
        self.iterations = iterations
        self.W = None
        self.b = None
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
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
                self.loss = elastic_net_loss(self.y, y_pred, w_i, self.lambda_1, self.lambda_2)

                lasso_term = np.sign(w_i) * self.lambda_1
                ridge_term = 2 * self.lambda_2 * w_i

                dw = -2/m * np.dot(self.x.T ,(self.y - y_pred)) + lasso_term + ridge_term
                db = -2/m * (np.sum(y - y_pred))
                
                self.W -= self.lr * dw
                self.b -= self.lr * db

            if i%20 == 0:
                print(f"Loss {i}: {self.loss}")
    
    def predict(self, instance: np.ndarray) -> int:
        y_pred = np.dot(instance, self.W) + self.b
        return y_pred