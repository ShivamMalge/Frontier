"""Domain defaults for the forecasting + optimization pipeline.

Framework-free on purpose: the numerical core (Layer*) imports this module, and
``app/settings.py`` layers environment-variable overrides on top of it.
"""

TECH_LIST = [
    'AAPL', 'GOOG', 'MSFT', 'ADBE', 'INTC', 'AMD',
    'LMT', 'NOC', 'BA', 'PFE', 'MRK', 'JNJ',
    'JPM', 'GS', 'MS', 'BLK', 'C', 'WFC'
]

START_DATE = "2010-01-01"
END_DATE = "2024-01-01"

RISK_FREE_RATE = 0.0      # annualised, as a decimal (0.04 == 4%)
LOOKBACK_WINDOW = 60      # LSTM sequence length
TRAIN_SPLIT = 2 / 3
LSTM_EPOCHS = 10
LSTM_BATCH_SIZE = 64

# Annualisation factor. Previously hardcoded as a literal 252 in
# Layer2_Optimization/mean_variance.py and Layer3_Portfolio_Generation/
# portfolio_compare.py; centralised here so the two can never disagree.
TRADING_DAYS_PER_YEAR = 252

# Minimum observations required before an LSTM will be trained on a series.
MIN_OBSERVATIONS = 200
