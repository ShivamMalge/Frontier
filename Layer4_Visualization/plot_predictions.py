# Layer4_Visualization/plot_predictions.py

import matplotlib.pyplot as plt

def plot_stock_prediction(dates, train_end_idx, actual_series, pred_series, symbol):
    plt.figure(figsize=(14,6))
    plt.title(f"{symbol} - LSTM Model")
    plt.xlabel("Date")
    plt.ylabel("Adj Close")
    plt.plot(dates[:train_end_idx], actual_series[:train_end_idx], label="Train")
    plt.plot(dates[train_end_idx:], actual_series[train_end_idx:], label="Val")
    plt.plot(dates[-len(pred_series):], pred_series, label="Predictions")
    plt.legend()
    plt.grid(True)
    plt.show()
