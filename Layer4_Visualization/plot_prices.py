# Layer4_Visualization/plot_prices.py

import matplotlib.pyplot as plt

def plot_prices(closing_df):
    plt.figure(figsize=(14, 6))
    for c in closing_df.columns:
        plt.plot(closing_df.index, closing_df[c], alpha=0.5)
    plt.title("Stock Prices")
    plt.xlabel("Date")
    plt.ylabel("Adj Close")
    plt.grid(True)
    plt.show()
