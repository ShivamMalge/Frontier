# Layer2_Optimization/gmv.py

import numpy as np


def gmv_weights(cov_matrix):
    inv_cov = np.linalg.pinv(cov_matrix)
    ones = np.ones(cov_matrix.shape[0])
    w = inv_cov @ ones
    w = w / w.sum()
    return w
