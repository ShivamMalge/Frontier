# Layer2_Optimization/constraints.py

"""Portfolio constraints, expressed once and translated for cvxpy.

This is the practical reason for moving off ``scipy.optimize.minimize``. Adding a
sector cap or a turnover limit to an SLSQP formulation means hand-writing another
callback and hoping it converges; here each one is a linear inequality that the
solver either satisfies exactly or declares infeasible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Constraints:
    """Convex constraints on a fully-invested portfolio.

    ``weights`` always sum to 1. Everything else is optional.
    """

    #: Per-asset bounds. A negative lower bound permits short positions.
    min_weight: float = 0.0
    max_weight: float = 1.0

    #: Cap on gross exposure, ``sum |w|``. 1.0 forbids shorting outright; 1.3 is a
    #: 130/30 mandate. Only meaningful when ``min_weight`` is negative.
    max_leverage: float | None = None

    #: Caps on named groups of assets, ``{"tech": (["AAPL", "MSFT"], 0.4)}``.
    group_caps: dict[str, tuple[list[str], float]] = field(default_factory=dict)

    #: Floors on named groups, same shape as ``group_caps``.
    group_floors: dict[str, tuple[list[str], float]] = field(default_factory=dict)

    #: Limit on ``sum |w - w_previous|``. Requires ``previous_weights``.
    max_turnover: float | None = None
    previous_weights: dict[str, float] = field(default_factory=dict)

    def validate(self, assets: list[str]) -> list[str]:
        """Return human-readable problems, without raising.

        Catching a contradiction here produces a better message than letting the
        solver report a generic infeasibility.
        """
        problems: list[str] = []
        n = len(assets)

        if self.min_weight > self.max_weight:
            problems.append(f"min_weight {self.min_weight} exceeds max_weight {self.max_weight}")
        if self.max_weight * n < 1.0 - 1e-9:
            problems.append(
                f"max_weight {self.max_weight} across {n} assets caps the portfolio at "
                f"{self.max_weight * n:.2f}, which cannot reach 1.0"
            )
        if self.min_weight * n > 1.0 + 1e-9:
            problems.append(
                f"min_weight {self.min_weight} across {n} assets forces at least "
                f"{self.min_weight * n:.2f}, which exceeds 1.0"
            )
        if self.max_leverage is not None and self.max_leverage < 1.0 - 1e-9:
            problems.append(
                f"max_leverage {self.max_leverage} is below 1.0, which a fully-invested "
                "portfolio cannot satisfy"
            )
        if self.max_turnover is not None and not self.previous_weights:
            problems.append("max_turnover requires previous_weights")

        known = set(assets)
        for label, (members, _cap) in {**self.group_caps, **self.group_floors}.items():
            unknown = sorted(set(members) - known)
            if unknown:
                problems.append(f"group '{label}' names unknown assets: {', '.join(unknown)}")

        for label, (_members, cap) in self.group_caps.items():
            floor = self.group_floors.get(label)
            if floor and floor[1] > cap:
                problems.append(f"group '{label}' floor {floor[1]} exceeds its cap {cap}")

        return problems

    def is_long_only(self) -> bool:
        return self.min_weight >= 0.0

    def previous_vector(self, assets: list[str]) -> np.ndarray | None:
        """Previous weights aligned to ``assets``; absent names count as zero."""
        if not self.previous_weights:
            return None
        return np.array([float(self.previous_weights.get(a, 0.0)) for a in assets])

    def cvxpy_constraints(self, w, assets: list[str]) -> list:
        """Translate into a list of cvxpy constraints for variable ``w``."""
        import cvxpy as cp

        index = {asset: position for position, asset in enumerate(assets)}
        constraints = [cp.sum(w) == 1.0, w >= self.min_weight, w <= self.max_weight]

        if self.max_leverage is not None:
            constraints.append(cp.norm1(w) <= self.max_leverage)

        for members, cap in self.group_caps.values():
            picks = [index[m] for m in members if m in index]
            if picks:
                constraints.append(cp.sum(w[picks]) <= cap)

        for members, floor in self.group_floors.values():
            picks = [index[m] for m in members if m in index]
            if picks:
                constraints.append(cp.sum(w[picks]) >= floor)

        previous = self.previous_vector(assets)
        if self.max_turnover is not None and previous is not None:
            constraints.append(cp.norm1(w - previous) <= self.max_turnover)

        return constraints


#: Long-only, fully invested, no other limits. Matches the pre-Phase-4 behaviour.
DEFAULT = Constraints()
