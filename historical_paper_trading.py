"""
Historical Paper Trading Bot (BTCUSDT, RSI & LSTM)
--------------------------------------------------
- Loads BTCUSDT data from CSV
- Starts simulation from 2025-01-01
- RSI strategy: Buy if RSI < 30, Sell if RSI > 70
- LSTM strategy: Train on data before 2025-01-01, predict next close, Buy if pred > current, Sell if pred < current
- Tracks and prints trades and portfolio value

Requirements:
    pip install pandas numpy scikit-learn tensorflow pandas-ta matplotlib
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Config
CSV_PATH = 'E:/TRADE/BTCUSDT_binance_historical_data.csv'
START_DATE = '2022-01-01'
INITIAL_BALANCE = 1000  # USDT
TRADE_SIZE = 0.001  # BTC per trade
N_LAGS = 24
EPOCHS = 50

# Load data
data = pd.read_csv(CSV_PATH, index_col='Date', parse_dates=True)

# Resample to 15-minute bars (fix deprecated 'T' usage)
data = data.resample('15min').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}).dropna()

# Calculate indicators
N_SMA = 20
N_EMA = 20
data['RSI_14'] = ta.rsi(data['Close'], length=14)
data['SMA_50'] = data['Close'].rolling(window=N_SMA).mean()
data['EMA_50'] = data['Close'].ewm(span=N_EMA, adjust=False).mean()

# Prepare LSTM features (use only Close for simplicity)
series = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series)

# Create lagged features for LSTM
X = []
y = []
for i in range(N_LAGS, len(series_scaled)):
    X.append(series_scaled[i-N_LAGS:i, 0])
    y.append(series_scaled[i, 0])
X = np.array(X)
y = np.array(y)

# Find start index for simulation
start_idx_data = data.index.get_indexer([START_DATE])[0]
start_idx = start_idx_data - N_LAGS  # Align with X/y arrays

# Train LSTM on data before START_DATE
X_train = X[:start_idx]
y_train = y[:start_idx]
X_test = X[start_idx:]
y_test = y[start_idx:]

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
print('Training LSTM model...')
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', input_shape=(N_LAGS, 1)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, verbose=1)

# Initialize portfolios
balance = INITIAL_BALANCE
btc_balance = 0
position_rsi = 0
position_lstm = 0
position_sma = 0
position_ema = 0
trade_log_rsi = []
trade_log_lstm = []
trade_log_sma = []
trade_log_ema = []

print(f"Starting historical paper trading from {START_DATE} on 30-minute timeframe...")

# Track portfolio value over time
portfolio_history = []

# Simulation loop
for i in range(start_idx + N_LAGS, len(data)):
    price = data['Close'].iloc[i]
    timestamp = data.index[i]
    rsi = data['RSI_14'].iloc[i]
    sma = data['SMA_50'].iloc[i]
    ema = data['EMA_50'].iloc[i]
    # RSI strategy
    signal_rsi = 0
    if rsi < 30:
        signal_rsi = 1  # Buy
    elif rsi > 70:
        signal_rsi = -1 # Sell
    # SMA strategy
    signal_sma = 0
    if price > sma:
        signal_sma = 1  # Buy
    elif price < sma:
        signal_sma = -1 # Sell
    # EMA strategy
    signal_ema = 0
    if price > ema:
        signal_ema = 1  # Buy
    elif price < ema:
        signal_ema = -1 # Sell
    # LSTM prediction (use previous N_LAGS closes)
    if i-N_LAGS < 0:
        continue
    last_input = series_scaled[i-N_LAGS:i, 0].reshape(1, N_LAGS, 1)
    pred_scaled = model.predict(last_input, verbose=0)[0, 0]
    pred = scaler.inverse_transform([[pred_scaled]])[0, 0]
    signal_lstm = 0
    if pred > price:
        signal_lstm = 1  # Buy
    elif pred < price:
        signal_lstm = -1 # Sell
    # Simulate RSI trades
    if signal_rsi == 1 and position_rsi == 0:
        btc_bought = TRADE_SIZE
        cost = btc_bought * price
        if balance >= cost:
            balance -= cost
            btc_balance += btc_bought
            position_rsi = 1
            trade_log_rsi.append((timestamp, 'BUY', price, btc_bought, balance, btc_balance))
            print(f"[RSI][{timestamp}] BUY {btc_bought} BTC at {price:.2f} | USDT: {balance:.2f} | BTC: {btc_balance:.4f}")
    elif signal_rsi == -1 and position_rsi == 1:
        btc_sold = btc_balance
        proceeds = btc_sold * price
        balance += proceeds
        btc_balance = 0
        position_rsi = 0
        trade_log_rsi.append((timestamp, 'SELL', price, btc_sold, balance, btc_balance))
        print(f"[RSI][{timestamp}] SELL {btc_sold} BTC at {price:.2f} | USDT: {balance:.2f} | BTC: {btc_balance:.4f}")
    # Simulate SMA trades
    if signal_sma == 1 and position_sma == 0:
        btc_bought = TRADE_SIZE
        cost = btc_bought * price
        if balance >= cost:
            balance -= cost
            btc_balance += btc_bought
            position_sma = 1
            trade_log_sma.append((timestamp, 'BUY', price, btc_bought, balance, btc_balance))
            print(f"[SMA][{timestamp}] BUY {btc_bought} BTC at {price:.2f} | USDT: {balance:.2f} | BTC: {btc_balance:.4f}")
    elif signal_sma == -1 and position_sma == 1:
        btc_sold = btc_balance
        proceeds = btc_sold * price
        balance += proceeds
        btc_balance = 0
        position_sma = 0
        trade_log_sma.append((timestamp, 'SELL', price, btc_sold, balance, btc_balance))
        print(f"[SMA][{timestamp}] SELL {btc_sold} BTC at {price:.2f} | USDT: {balance:.2f} | BTC: {btc_balance:.4f}")
    # Simulate EMA trades
    if signal_ema == 1 and position_ema == 0:
        btc_bought = TRADE_SIZE
        cost = btc_bought * price
        if balance >= cost:
            balance -= cost
            btc_balance += btc_bought
            position_ema = 1
            trade_log_ema.append((timestamp, 'BUY', price, btc_bought, balance, btc_balance))
            print(f"[EMA][{timestamp}] BUY {btc_bought} BTC at {price:.2f} | USDT: {balance:.2f} | BTC: {btc_balance:.4f}")
    elif signal_ema == -1 and position_ema == 1:
        btc_sold = btc_balance
        proceeds = btc_sold * price
        balance += proceeds
        btc_balance = 0
        position_ema = 0
        trade_log_ema.append((timestamp, 'SELL', price, btc_sold, balance, btc_balance))
        print(f"[EMA][{timestamp}] SELL {btc_sold} BTC at {price:.2f} | USDT: {balance:.2f} | BTC: {btc_balance:.4f}")
    # Simulate LSTM trades
    if signal_lstm == 1 and position_lstm == 0:
        btc_bought = TRADE_SIZE
        cost = btc_bought * price
        if balance >= cost:
            balance -= cost
            btc_balance += btc_bought
            position_lstm = 1
            trade_log_lstm.append((timestamp, 'BUY', price, btc_bought, balance, btc_balance))
            print(f"[LSTM][{timestamp}] BUY {btc_bought} BTC at {price:.2f} | USDT: {balance:.2f} | BTC: {btc_balance:.4f}")
    elif signal_lstm == -1 and position_lstm == 1:
        btc_sold = btc_balance
        proceeds = btc_sold * price
        balance += proceeds
        btc_balance = 0
        position_lstm = 0
        trade_log_lstm.append((timestamp, 'SELL', price, btc_sold, balance, btc_balance))
        print(f"[LSTM][{timestamp}] SELL {btc_sold} BTC at {price:.2f} | USDT: {balance:.2f} | BTC: {btc_balance:.4f}")
    # Print portfolio value
    portfolio_value = balance + btc_balance * price
    print(f"[{timestamp}] Portfolio Value: {portfolio_value:.2f} USDT\n")
    portfolio_history.append((timestamp, portfolio_value))

# Save trade logs and portfolio history to CSV files
pd.DataFrame(trade_log_rsi, columns=['timestamp','side','price','amount','usdt','btc']).to_csv('trades_rsi.csv', index=False)
pd.DataFrame(trade_log_sma, columns=['timestamp','side','price','amount','usdt','btc']).to_csv('trades_sma.csv', index=False)
pd.DataFrame(trade_log_ema, columns=['timestamp','side','price','amount','usdt','btc']).to_csv('trades_ema.csv', index=False)
pd.DataFrame(trade_log_lstm, columns=['timestamp','side','price','amount','usdt','btc']).to_csv('trades_lstm.csv', index=False)
pd.DataFrame(portfolio_history, columns=['timestamp','portfolio_value']).to_csv('portfolio_history.csv', index=False)
print('Trade logs and portfolio history saved to CSV files.') 