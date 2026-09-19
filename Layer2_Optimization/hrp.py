# Layer2_Optimization/hrp.py

"""Hierarchical Risk Parity (Lopez de Prado, 2016).

Three stages: build a correlation-distance linkage, quasi-diagonalise it so
correlated assets sit adjacent, then split capital by recursive bisection
weighted by inverse cluster variance.

Phase 4 of the migration replaces this module with Riskfolio-Lib, which also
brings HERC and the Gerber-based variants. Until then the implementation here is
correct rather than merely present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def correl_dist(corr: np.ndarray) -> np.ndarray:
    """Lopez de Prado's correlation distance: sqrt(0.5 * (1 - rho))."""
    return np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))


def get_quasi_diag(link: np.ndarray) -> list[int]:
    """Return original item indices ordered so that the linkage is quasi-diagonal.

    The cluster/item threshold is ``link[-1, 3]`` -- the number of original
    observations -- and *not* ``link.shape[0]``, which is one smaller. Using the
    row count treats the highest-numbered real asset as a cluster and expands it
    forever; this function previously looped indefinitely for that reason.
    """
    link = link.astype(int)
    num_items = int(link[-1, 3])

    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    while sort_ix.max() >= num_items:
        # Open a gap after every entry so each cluster can expand in place and
        # preserve ordering.
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        clusters = sort_ix[sort_ix >= num_items]
        positions = clusters.index
        rows = clusters.to_numpy() - num_items

        sort_ix[positions] = link[rows, 0]
        right = pd.Series(link[rows, 1], index=positions + 1)
        sort_ix = pd.concat([sort_ix, right]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])

    return sort_ix.astype(int).tolist()


def inverse_variance_weights(cov: np.ndarray) -> np.ndarray:
    """Inverse-variance portfolio over a covariance sub-matrix."""
    inv_var = 1.0 / np.diag(cov)
    return inv_var / inv_var.sum()


def get_cluster_var(cov: np.ndarray, items: list[int]) -> float:
    """Variance of a cluster held at its inverse-variance weights.

    The algorithm specifies inverse-variance weighting within the cluster; an
    equal-weight stand-in gives a different (and not-HRP) allocation.
    """
    sub_cov = cov[np.ix_(items, items)]
    w = inverse_variance_weights(sub_cov)
    return float(w @ sub_cov @ w)


def hrp_allocation(cov: np.ndarray) -> np.ndarray:
    """Long-only HRP weights, ordered to match the rows/columns of ``cov``."""
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.ones(1)

    std = np.sqrt(np.diag(cov))
    corr = np.nan_to_num(cov / np.outer(std, std), nan=0.0)
    np.fill_diagonal(corr, 1.0)

    link = linkage(squareform(correl_dist(corr), checks=False), method="single")
    sort_ix = get_quasi_diag(link)

    weights = pd.Series(1.0, index=sort_ix)
    clusters: list[list[int]] = [sort_ix]

    while clusters:
        cluster = clusters.pop(0)
        if len(cluster) <= 1:
            continue
        split = len(cluster) // 2
        left, right = cluster[:split], cluster[split:]

        left_var = get_cluster_var(cov, left)
        right_var = get_cluster_var(cov, right)
        alpha = 1.0 - left_var / (left_var + right_var + 1e-12)

        weights[left] *= alpha
        weights[right] *= 1.0 - alpha
        clusters.extend([left, right])

    w = weights.reindex(range(n)).fillna(0.0).to_numpy(dtype=float)
    total = w.sum()
    return w / total if total > 0 else np.full(n, 1.0 / n)
