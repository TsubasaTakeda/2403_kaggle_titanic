import numpy as np
from sklearn.linear_model import LogisticRegression

class Model():
    
    def __init__(self, params):
        self.params = params

    def train(self, X: np.ndarray, y: np.ndarray):
        self.model = LogisticRegression(**self.params)
        self.model.fit(X, y) 

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:,1]
        