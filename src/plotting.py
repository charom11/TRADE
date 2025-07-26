import matplotlib.pyplot as plt

def plot_signals(data, ticker):
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Close'], label='Close Price')
    if 'SMA_20' in data.columns:
        plt.plot(data.index, data['SMA_20'], label='SMA 20')
    buy_signals = data[data['Signal'] == 1]
    sell_signals = data[data['Signal'] == -1]
    plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='g', label='Buy Signal', alpha=1)
    plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='r', label='Sell Signal', alpha=1)
    plt.title(f'{ticker} Price and SMA Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

