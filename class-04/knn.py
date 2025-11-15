from collections import Counter
import numpy as np

class KNN:
    def __init__(self, k: int=9):
        self.k = k
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
    
    def predict(self, x: np.ndarray) -> int:
        distances = np.sqrt(np.sum(np.square(self.X - x), axis=1))
        # print(distances)
        
        indices = np.argpartition(distances, -self.k)[:self.k]
        # print(indices)
        
        groups = [self.y[i] for i in indices]
        # print(groups)
        
        result = Counter(groups).most_common()[0][0]
        # print(result)
        return result