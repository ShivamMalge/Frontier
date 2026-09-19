# Layer4_Visualization/plot_portfolio.py

import matplotlib.pyplot as plt

def plot_weights(weights_series):
    plt.figure(figsize=(10,6))
    weights_series.plot(kind='bar')
    plt.title("Final Portfolio Weights")
    plt.ylabel("Weight")
    plt.grid(True)
    plt.show()
