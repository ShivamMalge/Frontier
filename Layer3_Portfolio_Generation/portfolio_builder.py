# Layer3_Portfolio_Generation/portfolio_builder.py

"""Combine asset returns with strategy weights into portfolio return series."""

from __future__ import annotations

import pandas as pd


def build_portfolio_returns(returns_df: pd.DataFrame, weights_df: pd.DataFrame) -> pd.DataFrame:
    """One portfolio return series per column of ``weights_df``.

    ``weights_df`` is indexed by ticker; ``returns_df`` has one column per ticker.
    Alignment is by label, so column order does not matter.
    """
    shared = [ticker for ticker in weights_df.index if ticker in returns_df.columns]
    aligned_returns = returns_df[shared]
    aligned_weights = weights_df.loc[shared]

    # A matrix product aligned on tickers: (dates x assets) @ (assets x strategies).
    return aligned_returns.dot(aligned_weights)
