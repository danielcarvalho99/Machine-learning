from knn import KNNClassifier
from naive_bayes import NaiveBayesClassifier
from logistic_regression import LogisticRegressionClassifier
from svm import SVMClassifier
from neural_network import PerceptronClassifier, MLPClassifier
from decision_tree import DecisionTreeClassifier
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
    pcp = PerceptronClassifier()
    mlp = MLPClassifier(lr=4e-3, iterations=100, layers=np.array([3, 5, 1]))
    dt = DecisionTreeClassifier()
    

    for model in [nb, knn, lg, svm, pcp, mlp, dt]:
     model.train(x, y)
     pred = model.predict(instance)
     print(f"{model.__class__.__name__} predicted {pred}")

