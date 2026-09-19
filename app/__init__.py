"""Frontier: FastAPI service layer for LSTM forecasting and portfolio optimization.

Architecture (Phase 1 of the migration roadmap):

    app/routers/   HTTP surface -- thin, no business logic
    app/schemas/   Pydantic request/response contracts
    app/services/  adapters onto the numerical core
    app/jobs.py    async job seam (in-process now, RQ + Redis in Phase 2)

The numerical core is retained unchanged in spirit and lives in:

    Layer1_Preprocessing/  market data + returns
    Layer1_LSTM/           forecasting models and metrics
    Layer2_Optimization/   portfolio optimizers
    Layer3_Portfolio_Generation/  portfolio construction + performance
"""

__version__ = "0.1.0"
