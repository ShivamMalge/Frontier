# Layer3_Portfolio_Generation/portfolio_selector.py

"""Map a client's risk tolerance onto one of the computed strategies.

This is a positional pick along the volatility ranking: 0.0 selects the
lowest-volatility strategy, 1.0 the highest. It is a presentation convenience,
not a utility-maximising choice -- a genuine mapping would score strategies
against a risk-aversion parameter rather than index into a sorted list.
"""

from __future__ import annotations

import pandas as pd


def choose_portfolio_by_risk(perf_df: pd.DataFrame, risk_tolerance: float) -> str:
    if perf_df.empty:
        raise ValueError("no strategies to choose from")

    risk_tolerance = min(max(float(risk_tolerance), 0.0), 1.0)
    ranked = perf_df["Annual Vol"].sort_values().index.tolist()
    position = round(risk_tolerance * (len(ranked) - 1))
    return str(ranked[position])
