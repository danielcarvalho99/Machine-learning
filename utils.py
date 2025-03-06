import numpy as np

def compute_loss(y_true: np.ndarray, y_pred:np.ndarray) -> float:
        m = y_true.shape[0]
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = - (1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

def MSE_loss(y_true: np.ndarray, y_pred:np.ndarray) -> float:
        m = y_true.shape[0]
        loss = (1/m) * np.sum(np.sqrt((y_true - y_pred) ** 2))
        return loss

def lasso_loss(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray, lambda_val: float) -> float:
        m = y_true.shape[0]
        loss = (1/m) * (np.sum(np.sqrt((y_true - y_pred) ** 2)) + np.sum(lambda_val * np.absolute(w)))
        return loss

def ridge_loss(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray, lambda_val: float) -> float:
        m = y_true.shape[0]
        loss = (1/m) * (np.sum(np.sqrt((y_true - y_pred) ** 2)) + np.sum(lambda_val * (w**2)))
        return loss