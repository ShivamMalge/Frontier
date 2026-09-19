"""Adapters between the HTTP layer and the numerical core.

Routers depend on services; services depend on ``Layer*`` and ``utils``. Nothing
in the numerical core imports from ``app``.
"""
