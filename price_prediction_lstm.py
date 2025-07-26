"""
LSTM Price Prediction for BTCUSDT (OHLCV + Technical Indicators)
--------------------------------------------------------------
- Loads hourly BTCUSDT data from CSV
- Adds technical indicators (SMA_20, EMA_20, RSI_14, MACD, Bollinger Bands)
- Uses previous 24 hours of all features to predict next Close (recursive for 24 steps)
- Trains LSTM model (50 epochs)
- Forecasts next 24 hours
- Plots actual vs. predicted
- Prints MAE and RMSE

Requirements:
    pip install pandas numpy matplotlib scikit-learn tensorflow pandas-ta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pandas_ta as ta


# Load data (up to 2025-07-24)
csv_path = 'E:/TRADE/BTCUSDT_binance_historical_data.csv'
data = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
data = data.loc[:'2025-07-24 23:59:59']

# Add technical indicators
# SMA_20
data['SMA_20'] = data['Close'].rolling(window=20).mean()
# EMA_20
data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
# RSI_14
data['RSI_14'] = ta.rsi(data['Close'], length=14)
# MACD (use pandas-ta default: fast=12, slow=26, signal=9)
macd = ta.macd(data['Close'])
data['MACD'] = macd['MACD_12_26_9']
data['MACDs'] = macd['MACDs_12_26_9']
# Bollinger Bands
bbands = ta.bbands(data['Close'], length=20, std=2)
data['BBL_20_2.0'] = bbands['BBL_20_2.0']
data['BBU_20_2.0'] = bbands['BBU_20_2.0']

# Drop rows with NaN (from indicators)
data = data.dropna()

# Features to use
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'EMA_20', 'RSI_14', 'MACD', 'MACDs', 'BBL_20_2.0', 'BBU_20_2.0']
series = data[features].values

# Scale data
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series)

# Create lagged features (use previous 24 hours of all features to predict next Close)
N_LAGS = 24
X = []
y = []
for i in range(N_LAGS, len(series_scaled)):
    X.append(series_scaled[i-N_LAGS:i, :])
    y.append(series_scaled[i, features.index('Close')])
X = np.array(X)
y = np.array(y)

# Train/test split: last 24 for test, rest for train
X_train, X_test = X[:-24], X[-24:]
y_train, y_test = y[:-24], y[-24:]

# Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(N_LAGS, len(features))),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
print('Training LSTM model (50 epochs, OHLCV + technical indicators)...')
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Forecast next 24 hours (recursive)
preds_scaled = []
last_input = X_test[0].copy()
for i in range(24):
    pred = model.predict(last_input.reshape(1, N_LAGS, len(features)), verbose=0)[0, 0]
    preds_scaled.append(pred)
    # Shift window for next prediction
    last_input = np.roll(last_input, -1, axis=0)
    # For new row, copy all features from previous except set Close to pred
    new_row = last_input[-1].copy()
    new_row[features.index('Close')] = pred
    last_input[-1] = new_row

# Inverse scale predictions and test (only for Close)
preds_full = np.zeros((24, len(features)))
y_test_full = np.zeros((24, len(features)))
for i in range(24):
    preds_full[i, :] = series_scaled[-24+i, :]
    preds_full[i, features.index('Close')] = preds_scaled[i]
    y_test_full[i, :] = series_scaled[-24+i, :]

preds = scaler.inverse_transform(preds_full)[:, features.index('Close')]
y_test_inv = scaler.inverse_transform(y_test_full)[:, features.index('Close')]

# Plot
test_index = data.index[-24:]
plt.figure(figsize=(12,6))
plt.plot(data.index[-100:], data['Close'].values[-100:], label='Actual (last 100)')
plt.plot(test_index, preds, label='LSTM Forecast (OHLCV+Indicators)', marker='o')
plt.title('LSTM Forecast vs. Actual (BTCUSDT, 1H, OHLCV+Indicators, 50 epochs)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print error metrics
mae = mean_absolute_error(y_test_inv, preds)
rmse = np.sqrt(mean_squared_error(y_test_inv, preds))
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}') 