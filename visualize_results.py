"""
Visualize Paper Trading Results
------------------------------
- Plots portfolio value over time
- Plots buy/sell actions for RSI, SMA, EMA, LSTM strategies

Requirements:
    pip install pandas matplotlib
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load portfolio history
portfolio = pd.read_csv('portfolio_history.csv', parse_dates=['timestamp'])

# Load trade logs
trades_rsi = pd.read_csv('trades_rsi.csv', parse_dates=['timestamp'])
trades_sma = pd.read_csv('trades_sma.csv', parse_dates=['timestamp'])
trades_ema = pd.read_csv('trades_ema.csv', parse_dates=['timestamp'])
trades_lstm = pd.read_csv('trades_lstm.csv', parse_dates=['timestamp'])

# Plot portfolio value
tl = portfolio['timestamp']
pv = portfolio['portfolio_value']
plt.figure(figsize=(14, 7))
plt.plot(tl, pv, label='Portfolio Value', color='black')
plt.title('Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (USDT)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Helper to plot trades
strategy_trades = {
    'RSI': trades_rsi,
    'SMA': trades_sma,
    'EMA': trades_ema,
    'LSTM': trades_lstm
}

for name, trades in strategy_trades.items():
    plt.figure(figsize=(14, 5))
    plt.plot(tl, pv, label='Portfolio Value', color='gray', alpha=0.5)
    buys = trades[trades['side'] == 'BUY']
    sells = trades[trades['side'] == 'SELL']
    plt.scatter(buys['timestamp'], buys['usdt'], marker='^', color='g', label='Buy', alpha=0.7)
    plt.scatter(sells['timestamp'], sells['usdt'], marker='v', color='r', label='Sell', alpha=0.7)
    plt.title(f'{name} Strategy: Buy/Sell Actions vs. Portfolio Value')
    plt.xlabel('Date')
    plt.ylabel('USDT Balance After Trade')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show() 