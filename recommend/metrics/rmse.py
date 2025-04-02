import numpy as np

def rmse(predictions, ground_truths):    
    return np.sqrt(np.mean((predictions - ground_truths) ** 2))
