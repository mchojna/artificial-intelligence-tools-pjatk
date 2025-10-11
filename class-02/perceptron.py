import numpy as np


class Perceptron:
    def __init__(self, input_dim: int, activation_function):
        self.weight = np.random.rand(input_dim)
        self.bias = np.random.random()
        self.activation_function = activation_function

    def predict(self, X) -> int:
        z = np.dot(self.weight.T, X) + self.bias
        y = self.activation_function(z)
        return y


def step_function(z: float) -> int:
    return 1 if z > 0 else 0


# def relu(z: float) -> float:
#     return z if z > 0 else 0


# def sigmoid(z: float) -> float:
#     return 1 / (1 + np.e ^ (-z))
