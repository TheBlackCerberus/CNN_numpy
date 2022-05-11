import numpy as np

class CategoricalCrossEntropy:
    @staticmethod
    def loss(y, y_pred):
        eps = np.finfo(float).eps
        cross_entropy = -np.sum(y * np.log(y_pred + eps))
        return cross_entropy

    @staticmethod
    def grad(y, y_pred):
        return y_pred - y