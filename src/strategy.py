import pandas as pd
import numpy as np
from typing import List

class Strategy:
    """
    Base class for trading strategies.
    """
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("generate_signals must be implemented by subclasses.")
    
    def calculate_volatility(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate rolling volatility for dynamic thresholds."""
        return data['Close'].rolling(window=window).std()
    
    def calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range for position sizing."""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()

class SimpleMovingAverageStrategy(Strategy):
    """
    Simple Moving Average Crossover Strategy.
    Buy when Close > SMA, Sell when Close < SMA.
    """
    def __init__(self, window: int = 20):
        self.window = window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = data.copy()
        sma_col = f'SMA_{self.window}'
        signals[sma_col] = signals['Close'].rolling(window=self.window).mean()
        signals['Signal'] = 0
        signals.loc[signals['Close'] > signals[sma_col], 'Signal'] = 1  # Buy
        signals.loc[signals['Close'] < signals[sma_col], 'Signal'] = -1 # Sell
        return signals

class ExponentialMovingAverageStrategy(Strategy):
    """
    Exponential Moving Average (EMA) Crossover Strategy.
    Buy when Close > EMA, Sell when Close < EMA.
    """
    def __init__(self, window: int = 20):
        self.window = window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = data.copy()
        ema_col = f'EMA_{self.window}'
        signals[ema_col] = signals['Close'].ewm(span=self.window, adjust=False).mean()
        signals['Signal'] = 0
        signals.loc[signals['Close'] > signals[ema_col], 'Signal'] = 1  # Buy
        signals.loc[signals['Close'] < signals[ema_col], 'Signal'] = -1 # Sell
        return signals

class RSIStrategy(Strategy):
    """
    Advanced RSI Strategy with dynamic thresholds based on volatility.
    """
    def __init__(self, base_oversold: int = 30, base_overbought: int = 70, volatility_factor: float = 0.5):
        self.base_oversold = base_oversold
        self.base_overbought = base_overbought
        self.volatility_factor = volatility_factor
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = data.copy()
        signals['Signal'] = 0
        
        if 'RSI_14' in signals.columns:
            # Calculate volatility for dynamic thresholds
            volatility = self.calculate_volatility(data)
            volatility_normalized = (volatility - volatility.rolling(100).min()) / (volatility.rolling(100).max() - volatility.rolling(100).min())
            
            # Dynamic thresholds
            dynamic_oversold = self.base_oversold - (volatility_normalized * self.volatility_factor * 10)
            dynamic_overbought = self.base_overbought + (volatility_normalized * self.volatility_factor * 10)
            
            signals.loc[signals['RSI_14'] < dynamic_oversold, 'Signal'] = 1  # Buy
            signals.loc[signals['RSI_14'] > dynamic_overbought, 'Signal'] = -1 # Sell
            
            # Store thresholds for analysis
            signals['Dynamic_Oversold'] = dynamic_oversold
            signals['Dynamic_Overbought'] = dynamic_overbought
        
        return signals

class BollingerBandsStrategy(Strategy):
    """
    Bollinger Bands Strategy (window=20, k=2).
    Buy when Close < Lower Band, Sell when Close > Upper Band.
    """
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = data.copy()
        signals['Signal'] = 0
        if 'BBL_20_2.0' in signals.columns and 'BBU_20_2.0' in signals.columns:
            signals.loc[signals['Close'] < signals['BBL_20_2.0'], 'Signal'] = 1  # Buy
            signals.loc[signals['Close'] > signals['BBU_20_2.0'], 'Signal'] = -1 # Sell
        return signals

class MACDStrategy(Strategy):
    """
    Advanced MACD Strategy with histogram analysis and trend filtering.
    """
    def __init__(self, histogram_threshold: float = 0.1, trend_filter: bool = True):
        self.histogram_threshold = histogram_threshold
        self.trend_filter = trend_filter
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = data.copy()
        signals['Signal'] = 0
        
        if 'MACD_12_26_9' in signals.columns and 'MACDs_12_26_9' in signals.columns:
            macd = signals['MACD_12_26_9']
            macd_signal = signals['MACDs_12_26_9']
            
            # Calculate MACD histogram
            histogram = macd - macd_signal
            signals['MACD_Histogram'] = histogram
            
            # Trend filter using 50-period SMA
            if self.trend_filter and 'SMA_50' in signals.columns:
                trend_up = signals['Close'] > signals['SMA_50']
                trend_down = signals['Close'] < signals['SMA_50']
            else:
                trend_up = True
                trend_down = True
            
            # Enhanced signals with histogram confirmation
            buy_condition = (
                (macd > macd_signal) & 
                (macd.shift(1) <= macd_signal.shift(1)) &  # Crossover
                (histogram > self.histogram_threshold) &  # Strong momentum
                trend_up
            )
            
            sell_condition = (
                (macd < macd_signal) & 
                (macd.shift(1) >= macd_signal.shift(1)) &  # Crossunder
                (histogram < -self.histogram_threshold) &  # Strong momentum
                trend_down
            )
            
            signals.loc[buy_condition, 'Signal'] = 1
            signals.loc[sell_condition, 'Signal'] = -1
        
        return signals

class BollingerBandsWithTrendStrategy(Strategy):
    """
    Enhanced Bollinger Bands Strategy with trend filtering.
    """
    def __init__(self, bb_window: int = 20, bb_std: float = 2.0, trend_window: int = 50):
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.trend_window = trend_window
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = data.copy()
        signals['Signal'] = 0
        
        # Calculate Bollinger Bands
        sma = signals['Close'].rolling(window=self.bb_window).mean()
        std = signals['Close'].rolling(window=self.bb_window).std()
        signals['BB_Upper'] = sma + (std * self.bb_std)
        signals['BB_Lower'] = sma - (std * self.bb_std)
        signals['BB_Middle'] = sma
        
        # Trend filter
        trend_sma = signals['Close'].rolling(window=self.trend_window).mean()
        trend_up = signals['Close'] > trend_sma
        trend_down = signals['Close'] < trend_sma
        
        # Enhanced signals with trend confirmation
        buy_condition = (
            (signals['Close'] < signals['BB_Lower']) &  # Touch lower band
            trend_up &  # Uptrend
            (signals['Close'].shift(1) >= signals['BB_Lower'].shift(1))  # Just touched
        )
        
        sell_condition = (
            (signals['Close'] > signals['BB_Upper']) &  # Touch upper band
            trend_down &  # Downtrend
            (signals['Close'].shift(1) <= signals['BB_Upper'].shift(1))  # Just touched
        )
        
        signals.loc[buy_condition, 'Signal'] = 1
        signals.loc[sell_condition, 'Signal'] = -1
        
        return signals

class CompositeStrategy(Strategy):
    """
    Composite strategy combining multiple indicators for robust signals.
    """
    def __init__(self, rsi_weight: float = 0.3, macd_weight: float = 0.3, bb_weight: float = 0.2, trend_weight: float = 0.2):
        self.rsi_weight = rsi_weight
        self.macd_weight = macd_weight
        self.bb_weight = bb_weight
        self.trend_weight = trend_weight
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = data.copy()
        signals['Signal'] = 0
        signals['Composite_Score'] = 0
        
        # RSI component
        if 'RSI_14' in signals.columns:
            rsi_signal = 0
            rsi_signal = np.where(signals['RSI_14'] < 30, 1, rsi_signal)
            rsi_signal = np.where(signals['RSI_14'] > 70, -1, rsi_signal)
            signals['Composite_Score'] += rsi_signal * self.rsi_weight
        
        # MACD component
        if 'MACD_12_26_9' in signals.columns and 'MACDs_12_26_9' in signals.columns:
            macd = signals['MACD_12_26_9']
            macd_signal_line = signals['MACDs_12_26_9']
            macd_signal = np.where(macd > macd_signal_line, 1, -1)
            signals['Composite_Score'] += macd_signal * self.macd_weight
        
        # Bollinger Bands component
        if 'BBL_20_2.0' in signals.columns and 'BBU_20_2.0' in signals.columns:
            bb_signal = 0
            bb_signal = np.where(signals['Close'] < signals['BBL_20_2.0'], 1, bb_signal)
            bb_signal = np.where(signals['Close'] > signals['BBU_20_2.0'], -1, bb_signal)
            signals['Composite_Score'] += bb_signal * self.bb_weight
        
        # Trend component (SMA)
        sma_50 = signals['Close'].rolling(window=50).mean()
        trend_signal = np.where(signals['Close'] > sma_50, 1, -1)
        signals['Composite_Score'] += trend_signal * self.trend_weight
        
        # Generate final signals based on composite score
        signals.loc[signals['Composite_Score'] > 0.5, 'Signal'] = 1
        signals.loc[signals['Composite_Score'] < -0.5, 'Signal'] = -1
        
        return signals

class MeanReversionStrategy(Strategy):
    """
    Mean reversion strategy using Z-score and RSI.
    """
    def __init__(self, lookback: int = 20, z_threshold: float = 2.0, rsi_period: int = 14):
        self.lookback = lookback
        self.z_threshold = z_threshold
        self.rsi_period = rsi_period
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = data.copy()
        signals['Signal'] = 0
        
        # Calculate Z-score
        mean = signals['Close'].rolling(window=self.lookback).mean()
        std = signals['Close'].rolling(window=self.lookback).std()
        signals['Z_Score'] = (signals['Close'] - mean) / std
        
        # Use RSI for confirmation
        if 'RSI_14' in signals.columns:
            # Buy when price is significantly below mean AND RSI is oversold
            buy_condition = (
                (signals['Z_Score'] < -self.z_threshold) &
                (signals['RSI_14'] < 35)
            )
            
            # Sell when price is significantly above mean AND RSI is overbought
            sell_condition = (
                (signals['Z_Score'] > self.z_threshold) &
                (signals['RSI_14'] > 65)
            )
            
            signals.loc[buy_condition, 'Signal'] = 1
            signals.loc[sell_condition, 'Signal'] = -1
        
        return signals

class BreakoutStrategy(Strategy):
    """
    Breakout strategy using price channels and volume confirmation.
    """
    def __init__(self, channel_period: int = 20, volume_threshold: float = 1.5):
        self.channel_period = channel_period
        self.volume_threshold = volume_threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = data.copy()
        signals['Signal'] = 0
        
        # Calculate price channels
        signals['Channel_High'] = signals['High'].rolling(window=self.channel_period).max()
        signals['Channel_Low'] = signals['Low'].rolling(window=self.channel_period).min()
        
        # Volume filter
        avg_volume = signals['Volume'].rolling(window=self.channel_period).mean()
        high_volume = signals['Volume'] > avg_volume * self.volume_threshold
        
        # Breakout signals
        breakout_up = (
            (signals['Close'] > signals['Channel_High'].shift(1)) &
            high_volume
        )
        
        breakout_down = (
            (signals['Close'] < signals['Channel_Low'].shift(1)) &
            high_volume
        )
        
        signals.loc[breakout_up, 'Signal'] = 1
        signals.loc[breakout_down, 'Signal'] = -1
        
        return signals
