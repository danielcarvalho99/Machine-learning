import numpy as np
from functions import unit_step, ReLU, sigmoid
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

    def forward(self, x: np.ndarray):
        n_layers = self.layers.shape[0]
        for idx in range(n_layers - 1): 
            z = np.dot(x, self.W[idx]) + self.b[idx] 

            if idx == n_layers - 2: 
                x = sigmoid(z)
            else: 
                x = ReLU(z)

        return x  


    def train(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y
        n_layers = self.layers.shape[0]

        for i in range(n_layers - 1):
            self.W.append(np.random.randn(self.layers[i], self.layers[i+1]))
            self.b.append(np.zeros((1, self.layers[i+1])))

        print(self.iterations)
        for i in range(self.iterations):
            y_pred = self.forward(x)
            #loss = compute_loss(y, y_pred)


x = np.array([[1,0,1],
         [1,1,1],
         [0,0,0],
         [1,1,1]])
    
y = np.array([1,1,0,1])

mlp = MLPClassifier(layers=np.array([3, 8, 1]))
mlp.train(x, y)          