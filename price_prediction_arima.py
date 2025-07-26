"""
ARIMA Price Prediction for BTCUSDT
---------------------------------
- Loads hourly BTCUSDT data from CSV
- Fits ARIMA model to Close price
- Forecasts next 24 hours
- Plots actual vs. predicted
- Prints MAE and RMSE

Requirements:
    pip install pandas matplotlib statsmodels scikit-learn
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load data
csv_path = 'E:/TRADE/BTCUSDT_binance_historical_data.csv'
data = pd.read_csv(csv_path, index_col='Date', parse_dates=True)

# Use only the Close price
series = data['Close']

# Train/test split: last 24 hours for test, rest for train
train = series.iloc[:-24]
test = series.iloc[-24:]

# Fit ARIMA model (order can be tuned; (5,1,0) is a common start)
print('Fitting ARIMA model...')
model = ARIMA(train, order=(5,1,0))
model_fit = model.fit()

# Forecast next 24 hours
forecast = model_fit.forecast(steps=24)

# Plot
plt.figure(figsize=(12,6))
plt.plot(series.index[-100:], series.values[-100:], label='Actual (last 100)')
plt.plot(test.index, forecast, label='ARIMA Forecast', marker='o')
plt.title('ARIMA Forecast vs. Actual (BTCUSDT, 1H)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print error metrics
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}') 