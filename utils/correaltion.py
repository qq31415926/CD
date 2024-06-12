import numpy as np

def pearson_correlation(x, y):
    
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    
    numerator = np.sum((x - mean_x) * (y - mean_y))

    
    denominator = np.sqrt(np.sum((x - mean_x) ** 2) * np.sum((y - mean_y) ** 2))

    
    r = numerator / denominator

    return r
