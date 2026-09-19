# Layer4_Visualization/plot_efficient_frontier.py

import matplotlib.pyplot as plt

def plot_cumulative_portfolios(cum_df):
    plt.figure(figsize=(14,7))
    for c in cum_df.columns:
        plt.plot(cum_df.index, cum_df[c], label=c)
    plt.title("Cumulative Growth")
    plt.xlabel("Date")
    plt.ylabel("Growth")
    plt.legend()
    plt.grid(True)
    plt.show()
