import pandas as pd
import pandas_ta as ta

def add_indicators(data):
    """
    Calculates and adds comprehensive technical indicators for advanced trading strategies.

    Args:
        data (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume' columns.

    Returns:
        pd.DataFrame: DataFrame with added indicator columns.
    """
    print("Calculating comprehensive indicators...")

    # Basic indicators
    data.ta.rsi(append=True)
    data.ta.macd(append=True)
    data.ta.vwap(append=True)
    
    # Multiple SMA periods for different strategies
    for period in [9, 20, 50, 100, 200]:
        data[f'SMA_{period}'] = data['Close'].rolling(window=period).mean()
    
    # Multiple EMA periods
    for period in [9, 12, 20, 26, 50]:
        data[f'EMA_{period}'] = data['Close'].ewm(span=period, adjust=False).mean()
    
    # Bollinger Bands (window=20, k=2)
    sma_20 = data['Close'].rolling(window=20).mean()
    std_20 = data['Close'].rolling(window=20).std()
    data['BBL_20_2.0'] = sma_20 - 2 * std_20
    data['BBU_20_2.0'] = sma_20 + 2 * std_20
    data['BB_Middle_20'] = sma_20
    
    # Average True Range (ATR) for volatility measurement
    data.ta.atr(append=True)
    
    # Stochastic Oscillator
    data.ta.stoch(append=True)
    
    # Williams %R
    data.ta.willr(append=True)
    
    # Commodity Channel Index (CCI)
    data.ta.cci(append=True)
    
    # Money Flow Index (MFI)
    data.ta.mfi(append=True)
    
    # On Balance Volume (OBV)
    data.ta.obv(append=True)
    
    # Volume indicators
    data['Volume_SMA_20'] = data['Volume'].rolling(window=20).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA_20']
    
    # Price channels for breakout detection
    for period in [10, 20, 50]:
        data[f'High_Channel_{period}'] = data['High'].rolling(window=period).max()
        data[f'Low_Channel_{period}'] = data['Low'].rolling(window=period).min()
    
    # Volatility indicators
    data['Price_Volatility_20'] = data['Close'].rolling(window=20).std()
    data['Price_Volatility_50'] = data['Close'].rolling(window=50).std()
    
    # Rate of Change (ROC)
    for period in [5, 10, 20]:
        data[f'ROC_{period}'] = data['Close'].pct_change(periods=period) * 100
    
    # Average Directional Index (ADX)
    data.ta.adx(append=True)
    
    # Parabolic SAR
    data.ta.psar(append=True)
    
    # Fill NaN values with appropriate methods
    # Forward fill for most indicators
    data = data.ffill()
    # Remaining NaN values filled with 0
    data = data.fillna(0)

    print("Comprehensive indicators calculated and added.")
    return data
