# Layer2_Optimization/gerber.py

import numpy as np
import pandas as pd


def gerber_covariance(returns: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    R = returns.to_numpy()
    n, d = R.shape
    sign_matrix = np.sign(R)
    G = np.zeros((d, d))

    for i in range(d):
        for j in range(d):
            same_sign = (sign_matrix[:, i] * sign_matrix[:, j]) > 0
            prob = np.mean(same_sign)
            sij = (prob - threshold) / (1 - threshold)
            sij = np.clip(sij, 0, 1)
            G[i, j] = sij * np.std(R[:, i]) * np.std(R[:, j])

    return pd.DataFrame(G, index=returns.columns, columns=returns.columns)

def gerber_inverse_var_weights(cov_matrix: pd.DataFrame) -> pd.Series:
    inv_var = 1 / np.diag(cov_matrix)
    w = inv_var / inv_var.sum()
    return pd.Series(w, index=cov_matrix.index)
