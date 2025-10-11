import numpy as np
from perceptron import Perceptron


class Trainer:
    def __init__(self, epochs: int = 32, learning_rate: float = 0.001):
        self.epochs = epochs
        self.learning_rate = learning_rate

    def train(self, X: np.ndarray, y: np.ndarray, perceptron: Perceptron):
        for _ in range(self.epochs):
            for i in range(X.shape[0]):
                y_hat = perceptron.predict(X[i,])

                perceptron.weight += self.learning_rate * (y[i] - y_hat) * X[i,]
                perceptron.bias += self.learning_rate * (y[i] - y_hat)

    def predict(self, X: np.ndarray, y: np.ndarray, perceptron: Perceptron):
        predictions = np.zeros(y.shape)

        for i in range(X.shape[0]):
            y_hat = perceptron.predict(X[i,])
            predictions[i] = y_hat

        print(f"Actuals: {y}")
        print(f"Predictions: {predictions.astype(int)}")
        print(f"Accuracy: {np.mean(predictions.astype(int) == y)}")
