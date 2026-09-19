# Layer2_Optimization/gmv.py

"""Global minimum variance in closed form.

Superseded in Phase 4 and no longer called by the API, the CLI or the Streamlit
app -- ``optimizer_master`` now dispatches to ``convex`` (cvxpy) and
``riskfolio_strategies`` (Riskfolio-Lib). Kept because its tests encode the
defects found during the audit, which is worth preserving as documentation.
Safe to delete once that history is no longer useful.

Uses the pseudo-inverse, so short positions are permitted. Mathematically this is
minimum variance without a non-negativity constraint, not a distinct strategy.
"""
import numpy as np


def gmv_weights(cov_matrix):
    inv_cov = np.linalg.pinv(cov_matrix)
    ones = np.ones(cov_matrix.shape[0])
    w = inv_cov @ ones
    w = w / w.sum()
    return w
