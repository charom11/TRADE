"""
Evaluate All Strategies
----------------------
- Loads trade logs for all strategies
- Calculates final portfolio value and total return for each
- Prints a summary and highlights the best performer

Requirements:
    pip install pandas
"""

import pandas as pd

# Load trade logs
def load_trades(filename):
    try:
        return pd.read_csv(filename)
    except Exception:
        return pd.DataFrame(columns=['timestamp','side','price','amount','usdt','btc'])

strategies = ['RSI', 'SMA', 'EMA', 'LSTM', 'BBTrend', 'Composite', 'MeanReversion', 'Breakout']
results = {}

for strat in strategies:
    trades = load_trades(f'trades_{strat.lower()}.csv')
    if trades.empty:
        results[strat] = {'final_usdt': 0, 'final_btc': 0, 'final_value': 0, 'return_pct': -100}
        continue
    # Use last USDT and BTC balance in the log
    last = trades.iloc[-1]
    final_usdt = last['usdt']
    final_btc = last['btc']
    final_price = last['price']
    final_value = final_usdt + final_btc * final_price
    initial_usdt = trades.iloc[0]['usdt'] + trades.iloc[0]['btc'] * trades.iloc[0]['price']
    return_pct = 100 * (final_value - initial_usdt) / initial_usdt
    results[strat] = {
        'final_usdt': final_usdt,
        'final_btc': final_btc,
        'final_value': final_value,
        'return_pct': return_pct
    }

# Print summary
def print_summary(results):
    print("\nStrategy Performance Summary:")
    print(f"{'Strategy':<12} {'Final Value (USDT)':>20} {'Return (%)':>15}")
    for strat, res in results.items():
        print(f"{strat:<12} {res['final_value']:>20.2f} {res['return_pct']:>15.2f}")
    best = max(results.items(), key=lambda x: x[1]['final_value'])[0]
    print(f"\nMost effective strategy: {best} (Final Value: {results[best]['final_value']:.2f} USDT, Return: {results[best]['return_pct']:.2f}%)")
    print(f"\nRecommended: Use the {best} strategy for future trading.")

print_summary(results)
