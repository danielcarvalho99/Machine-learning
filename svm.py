import numpy as np

class SVMClassifier():
    def __init__(self, lr: float=1e-3, lambda_param: float=1e-2, iterations: int=100):
        self.W = None
        self.b = None
        self.lr = lr
        self.lambda_param = lambda_param
        self.iterations = iterations

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.y = y

        y_ = np.where(y <= 0, -1, 1)
        
        self.W = np.random.rand(x.shape[1])  
        self.b = 0

        for _ in range(self.iterations):
            for idx, x_i in enumerate(x):
                condition = y_[idx] * (np.dot(x_i, self.W) + self.b) >= 1
                if condition:
                    dw = 2 * self.lambda_param * self.W
                    db = 0
                else:
                    dw = 2 * self.lambda_param * self.W - np.dot(x_i, y_[idx])
                    db = y_[idx]
                
                self.W -= self.lr * dw
                self.b -= self.lr * db

    def predict(self, instance: np.ndarray):
        pred = np.dot(instance, self.W) - self.b
        return int(np.sign(pred))