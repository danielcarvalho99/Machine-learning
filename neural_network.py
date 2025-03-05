import numpy as np
from functions import *
from utils import compute_loss

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

    def predict(self, instance: np.ndarray) -> int:
         z = np.dot(instance, self.W) + self.b
         return unit_step(z)

class MLPClassifier():
    def __init__(self, lr: float = 1e-3, iterations: int = 5, layers: np.ndarray = None):
        self.W = []
        self.b = []
        self.lr = lr
        self.iterations = iterations
        self.layers = layers

        for i in range(self.layers.shape[0] - 1):
            self.W.append(np.random.randn(self.layers[i], self.layers[i + 1]))
            self.b.append(np.zeros((1, self.layers[i + 1])))

    def forward(self, x: np.ndarray):
        activations = [x]
        num_layers = self.layers.shape[0]

        for idx in range(num_layers - 1):
            z = np.dot(x, self.W[idx]) + self.b[idx]
            x = sigmoid(z) if idx == num_layers - 2 else ReLU(z)
            activations.append(x)

        return activations

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        m = y.shape[0]
        y = y.reshape(-1,1)

        for _ in range(self.iterations):
            activations = self.forward(x)
            y_pred = activations[-1]
            loss = compute_loss(y, y_pred)

            dz = y_pred - y

            for idx in range(len(self.layers) - 2, -1, -1):
                a_prev = activations[idx]

                dw = np.dot(a_prev.T, dz) / m  
                db = np.sum(dz, axis=0, keepdims=True) / m 

                self.W[idx] -= self.lr * dw
                self.b[idx] -= self.lr * db

                if idx > 0:
                    dz = np.dot(dz, self.W[idx].T) * ReLU_derivative(activations[idx])

    def predict(self, x: np.ndarray):
        y_pred = self.forward(x)
        return (y_pred[len(y_pred) - 1] >= 0.5).astype(int).flatten()[0]
