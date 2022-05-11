import numpy as np

class ReLu:
    def __init__(self):
        self.cache = {}

    def forward_pass(self, Z, save_cache=True):
        if save_cache:
            Z = self.cache["Z"]
        Z = np.max(0.0, Z)
        return Z

    def backward_pass(self, dA):
        Z = self.cache["Z"]
        dA * np.where(Z >= 0, 1, 0)

class Softmax:
    def __init__(self):
        self.cache = {}

    def forward_pass(self, Z, save_cache=True):
        if save_cache:
            Z = self.cache["Z"]
        e_x = np.exp(Z - np.max(Z))
        return e_x / np.sum(e_x, axis=0, keepdims=True)

    def backward_pass(self, dA):
        Z = self.cache["Z"]
        return Z * (1. - Z) * dA






