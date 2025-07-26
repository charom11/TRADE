"""
Real-Time Paper Trading Bot (Binance, BTCUSDT, SMA Example)
----------------------------------------------------------
- Fetches live BTCUSDT price from Binance using ccxt
- Runs a simple SMA strategy (can be replaced with any logic)
- Simulates trades and tracks virtual balance (no real orders)
- Prints trades and performance
- Now with error handling and debug prints

Requirements:
    pip install ccxt pandas numpy

Usage:
    python paper_trading_bot.py
"""

import ccxt
import pandas as pd
import numpy as np
import time

# Config
SYMBOL = 'BTC/USDT'
EXCHANGE = 'binance'
TIMEFRAME = '1m'  # 1-minute candles
SMA_WINDOW = 20   # SMA period
INITIAL_BALANCE = 10000  # USDT
TRADE_SIZE = 0.001  # BTC per trade
LOOP_INTERVAL = 60  # seconds

# Initialize exchange (no API keys needed for public data)
exchange = getattr(ccxt, EXCHANGE)()

# Virtual portfolio
balance = INITIAL_BALANCE
btc_balance = 0
position = 0  # 0 = no position, 1 = long
trade_log = []

print(f"Starting paper trading bot on {SYMBOL} ({TIMEFRAME}) with SMA({SMA_WINDOW})...")

while True:
    try:
        # Fetch latest candles
        ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=SMA_WINDOW+2)
        if not ohlcv or len(ohlcv) < SMA_WINDOW+1:
            print(f"[ERROR] Not enough data returned from exchange. Retrying in 10 seconds...")
            time.sleep(10)
            continue
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df['close'] = df['close'].astype(float)
        sma = df['close'].rolling(window=SMA_WINDOW).mean().iloc[-1]
        price = df['close'].iloc[-1]
        timestamp = pd.to_datetime(df['timestamp'].iloc[-1], unit='ms')

        # Simple SMA strategy: Buy if price > SMA, Sell if price < SMA
        signal = 0
        if price > sma:
            signal = 1  # Buy
        elif price < sma:
            signal = -1 # Sell

        # Simulate trades
        if signal == 1 and position == 0:
            # Buy
            btc_bought = TRADE_SIZE
            cost = btc_bought * price
            if balance >= cost:
                balance -= cost
                btc_balance += btc_bought
                position = 1
                trade_log.append((timestamp, 'BUY', price, btc_bought, balance, btc_balance))
                print(f"[{timestamp}] BUY {btc_bought} BTC at {price:.2f} | USDT: {balance:.2f} | BTC: {btc_balance:.4f}")
        elif signal == -1 and position == 1:
            # Sell
            btc_sold = btc_balance
            proceeds = btc_sold * price
            balance += proceeds
            btc_balance = 0
            position = 0
            trade_log.append((timestamp, 'SELL', price, btc_sold, balance, btc_balance))
            print(f"[{timestamp}] SELL {btc_sold} BTC at {price:.2f} | USDT: {balance:.2f} | BTC: {btc_balance:.4f}")
        else:
            print(f"[{timestamp}] HOLD | Price: {price:.2f} | SMA: {sma:.2f} | USDT: {balance:.2f} | BTC: {btc_balance:.4f}")

        # Print current portfolio value
        portfolio_value = balance + btc_balance * price
        print(f"Portfolio Value: {portfolio_value:.2f} USDT\n")

        # Wait for next loop
        time.sleep(LOOP_INTERVAL)
    except Exception as e:
        print(f"[ERROR] {e}. Retrying in 10 seconds...")
        time.sleep(10) 