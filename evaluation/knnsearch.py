from scipy.spatial.distance import cdist
import numpy as np

def knnsearch(A, B):
    dist = cdist(A, B)
    match = np.argmin(dist, axis=1)
    return match

