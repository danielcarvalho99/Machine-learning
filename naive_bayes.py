import numpy as np

class NaiveBayesClassifier():
    def __init__(self):
        pass

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.y = y

    def predict(self, instance: np.ndarray) -> int:
        max_prob = 0
        res = self.y[0]
        for outcome in set(self.y):
            stacked_array = np.column_stack((self.x,self.y))
        
            idx_filtered = np.where(stacked_array[:, -1] == outcome)
            x_filtered = self.x[idx_filtered]

            prob_outcome = 1 
            for i, val in enumerate(instance):
                count = np.count_nonzero(x_filtered[:, i] == val)
                prob_outcome *= (count / self.y.shape[0])
        
            if prob_outcome > max_prob:
                res = outcome

        return res