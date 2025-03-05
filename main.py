from knn import KNNClassifier
from naive_bayes import NaiveBayesClassifier
from logistic_regression import LogisticRegressionClassifier
from svm import SVMClassifier
import numpy as np


if __name__ == '__main__':

    x = np.array([[1,0,1],
         [1,1,1],
         [0,0,0],
         [1,1,1]])
    
    y = np.array([1,1,0,1])

    instance = np.array([1,1,1])

    nb = NaiveBayesClassifier()
    knn = KNNClassifier(3)
    lg = LogisticRegressionClassifier()
    svm = SVMClassifier()
    

    for model in [nb, knn, lg, svm]:
     model.train(x, y)
     pred = model.predict(instance)
     print(f"{model.__class__.__name__} predicted {pred}")

