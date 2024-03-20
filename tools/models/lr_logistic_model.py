import numpy as np
from sklearn.linear_model import LogisticRegression

class Model():
    
    def __init__(self, params):
        self.params = params

    def train(self, X: np.ndarray, y: np.ndarray):
        self.lr_model = LogisticRegression(**self.params)
        self.lr_model.fit(X, y) 

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.lr_model.predict_proba(X)[:,1]
        