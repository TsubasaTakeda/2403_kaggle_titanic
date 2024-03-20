import numpy as np

class Postprocessor():

    def __init__(self, params):
        self.params = params

    def transform(self, y: np.ndarray) -> np.ndarray:
        threshold = self.params['threshold']
        pred_y = np.where(y > threshold, 1, 0)
        return pred_y
