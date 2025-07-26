"""
Advanced Risk Management and Position Sizing Module
==================================================
Implements sophisticated risk controls and dynamic position sizing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class AdvancedRiskManager:
    """
    Advanced risk management with dynamic position sizing, 
    volatility-based stops, and portfolio heat controls
    """
    
    def __init__(self, 
                 max_portfolio_risk: float = 0.02,  # 2% max portfolio risk
                 max_single_position: float = 0.10,  # 10% max single position
                 volatility_lookback: int = 20,
                 atr_multiplier: float = 2.0):
        
        self.max_portfolio_risk = max_portfolio_risk
        self.max_single_position = max_single_position
        self.volatility_lookback = volatility_lookback
        self.atr_multiplier = atr_multiplier
        
    def calculate_position_size(self, 
                              account_balance: float,
                              entry_price: float,
                              stop_loss: float,
                              atr: float,
                              confidence: float = 1.0) -> float:
        """
        Calculate optimal position size using multiple methods
        """
        # Method 1: Fixed percentage risk
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            return 0
        
        fixed_risk_size = (account_balance * self.max_portfolio_risk) / risk_per_share
        
        # Method 2: ATR-based sizing
        atr_risk = atr * self.atr_multiplier
        atr_size = (account_balance * self.max_portfolio_risk) / atr_risk
        
        # Method 3: Volatility-adjusted sizing
        volatility_factor = min(atr / entry_price, 0.05)  # Cap at 5%
        vol_adjusted_size = fixed_risk_size * (1 - volatility_factor)
        
        # Method 4: Confidence-weighted sizing
        confidence_size = fixed_risk_size * confidence
        
        # Use the most conservative size
        position_size = min(fixed_risk_size, atr_size, vol_adjusted_size, confidence_size)
        
        # Apply maximum position limit
        max_position_value = account_balance * self.max_single_position
        max_shares = max_position_value / entry_price
        
        return min(position_size, max_shares)
    
    def calculate_dynamic_stops(self, data: pd.DataFrame, lookback: int = 20) -> Dict:
        """
        Calculate dynamic stop losses based on multiple methods
        """
        # ATR-based stops
        atr = self.calculate_atr(data, lookback)
        atr_stop_distance = atr * self.atr_multiplier
        
        # Volatility-based stops
        returns = data['Close'].pct_change()
        volatility = returns.rolling(lookback).std() * np.sqrt(252)
        vol_stop_distance = data['Close'] * volatility * 0.1  # 10% of volatility
        
        # Support/Resistance based stops
        support_levels = data['Low'].rolling(lookback).min()
        resistance_levels = data['High'].rolling(lookback).max()
        
        return {
            'atr_stop_long': data['Close'] - atr_stop_distance,
            'atr_stop_short': data['Close'] + atr_stop_distance,
            'vol_stop_long': data['Close'] - vol_stop_distance,
            'vol_stop_short': data['Close'] + vol_stop_distance,
            'support_stop': support_levels,
            'resistance_stop': resistance_levels
        }
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

class PortfolioHeatManager:
    """
    Manages overall portfolio heat and correlation risk
    """
    
    def __init__(self, max_heat: float = 0.20, max_correlated_positions: int = 3):
        self.max_heat = max_heat  # Maximum 20% portfolio at risk
        self.max_correlated_positions = max_correlated_positions
        self.active_positions = {}
        
    def calculate_portfolio_heat(self, positions: Dict) -> float:
        """Calculate current portfolio heat (total risk exposure)"""
        total_risk = sum(pos.get('risk_amount', 0) for pos in positions.values())
        portfolio_value = sum(pos.get('position_value', 0) for pos in positions.values())
        
        if portfolio_value == 0:
            return 0
        
        return total_risk / portfolio_value
    
    def can_add_position(self, new_position_risk: float, portfolio_value: float) -> bool:
        """Check if new position can be added without exceeding heat limits"""
        current_heat = self.calculate_portfolio_heat(self.active_positions)
        new_heat = new_position_risk / portfolio_value
        
        return (current_heat + new_heat) <= self.max_heat

class MarketRegimeDetector:
    """
    Detect market regimes to adjust strategy parameters
    """
    
    def __init__(self, lookback: int = 50):
        self.lookback = lookback
    
    def detect_regime(self, data: pd.DataFrame) -> Dict:
        """
        Detect current market regime: trending, ranging, volatile
        """
        # Trend strength
        sma_short = data['Close'].rolling(10).mean()
        sma_long = data['Close'].rolling(50).mean()
        trend_strength = abs(sma_short - sma_long) / sma_long
        
        # Volatility regime
        returns = data['Close'].pct_change()
        current_vol = returns.rolling(20).std()
        vol_regime = current_vol / returns.rolling(100).std()
        
        # Range detection
        high_range = data['High'].rolling(self.lookback).max()
        low_range = data['Low'].rolling(self.lookback).min()
        range_position = (data['Close'] - low_range) / (high_range - low_range)
        
        regime = {
            'trend_strength': trend_strength.iloc[-1] if len(trend_strength) > 0 else 0,
            'volatility_regime': vol_regime.iloc[-1] if len(vol_regime) > 0 else 1,
            'range_position': range_position.iloc[-1] if len(range_position) > 0 else 0.5,
            'is_trending': trend_strength.iloc[-1] > 0.02 if len(trend_strength) > 0 else False,
            'is_high_vol': vol_regime.iloc[-1] > 1.5 if len(vol_regime) > 0 else False,
            'is_ranging': trend_strength.iloc[-1] < 0.01 if len(trend_strength) > 0 else True
        }
        
        return regime
