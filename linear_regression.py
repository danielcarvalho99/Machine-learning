import numpy as np
from utils import MSE_loss

class LinearRegression():
    def __init__(self, lr: float=1e-3, iterations: int=2000):
        self.lr = lr
        self.iterations = iterations
        self.W = None
        self.b = None

    def train(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y.reshape(-1,1)
        m, n = self.x.shape

        self.W = np.random.rand(n, 1)
        self.b = np.zeros((1, 1))

        for _ in range(self.iterations):
            y_pred = np.dot(self.x, self.W) + self.b
            MSE_loss(self.y, y_pred)
        
            dw = -2/m * np.dot(self.x.T ,(self.y - y_pred))
            db = -2/m * (np.sum(y - y_pred))
            
            self.W -= self.lr * dw
            self.b -= self.lr * db
    
    def predict(self, instance: np.ndarray) -> int:
        y_pred = np.dot(instance, self.W) + self.b
        return y_pred

lr = LinearRegression()
x = np.array([[1],[2.1], [3.2], [4]])
y = np.array([3, 5.2, 7.2, 9.2])
lr.train(x, y)
print(lr.predict([1]))