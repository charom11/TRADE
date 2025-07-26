"""
Linear Regression Price Prediction for BTCUSDT
---------------------------------------------
- Loads hourly BTCUSDT data from CSV
- Uses lagged Close prices as features
- Fits Linear Regression model
- Forecasts the next 24 hours
- Plots actual vs. predicted
- Prints MAE and RMSE

Requirements:
    pip install pandas matplotlib scikit-learn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
csv_path = 'E:/TRADE/BTCUSDT_binance_historical_data.csv'
data = pd.read_csv(csv_path, index_col='Date', parse_dates=True)

# Use only the Close price
series = data['Close']

# Create lagged features (e.g., use previous 24 hours to predict next hour)
N_LAGS = 24
X = []
y = []
for i in range(N_LAGS, len(series)):
    X.append(series.values[i-N_LAGS:i])
    y.append(series.values[i])
X = np.array(X)
y = np.array(y)

# Train/test split: last 24 for test, rest for train
X_train, X_test = X[:-24], X[-24:]
y_train, y_test = y[:-24], y[-24:]

# Fit Linear Regression
print('Fitting Linear Regression model...')
model = LinearRegression()
model.fit(X_train, y_train)

# Forecast next 24 hours (recursive, using previous predictions)
preds = []
last_input = X_test[0].copy()
for i in range(24):
    pred = model.predict(last_input.reshape(1, -1))[0]
    preds.append(pred)
    # Shift window for next prediction
    last_input = np.roll(last_input, -1)
    last_input[-1] = pred

# Plot
test_index = series.index[-24:]
plt.figure(figsize=(12,6))
plt.plot(series.index[-100:], series.values[-100:], label='Actual (last 100)')
plt.plot(test_index, preds, label='Linear Regression Forecast', marker='o')
plt.title('Linear Regression Forecast vs. Actual (BTCUSDT, 1H)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print error metrics
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}') 