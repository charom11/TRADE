"""
Enhanced Adaptive Trading Strategies
====================================
Implements regime-aware strategies with dynamic parameter adjustment
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from .strategy import Strategy
from .risk_management import AdvancedRiskManager, MarketRegimeDetector

class AdaptiveSMAStrategy(Strategy):
    """
    SMA strategy that adapts parameters based on market regime
    """
    
    def __init__(self, base_window: int = 50, regime_detector=None):
        self.base_window = base_window
        self.regime_detector = regime_detector or MarketRegimeDetector()
        self.risk_manager = AdvancedRiskManager()
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = data.copy()
        signals['Signal'] = 0
        signals['Position_Size'] = 0
        signals['Stop_Loss'] = 0
        signals['Confidence'] = 0
        
        # Detect market regime
        regime = self.regime_detector.detect_regime(data)
        
        # Adapt window based on regime
        if regime['is_trending']:
            window = max(self.base_window - 10, 20)  # Shorter for trending
        elif regime['is_ranging']:
            window = self.base_window + 20  # Longer for ranging
        else:
            window = self.base_window
        
        # Calculate adaptive SMA
        signals['SMA'] = signals['Close'].rolling(window=window).mean()
        
        # Enhanced signal generation with confidence
        price_above_sma = signals['Close'] > signals['SMA']
        price_below_sma = signals['Close'] < signals['SMA']
        
        # Distance from SMA as confidence measure
        sma_distance = abs(signals['Close'] - signals['SMA']) / signals['SMA']
        confidence = np.clip(sma_distance * 10, 0.1, 1.0)  # Scale to 0.1-1.0
        
        # Generate signals with regime awareness
        for i in range(len(signals)):
            current_regime = regime  # In practice, would recalculate per bar
            
            if price_above_sma.iloc[i]:
                if current_regime['is_trending'] or not current_regime['is_high_vol']:
                    signals.iloc[i, signals.columns.get_loc('Signal')] = 1
                    signals.iloc[i, signals.columns.get_loc('Confidence')] = confidence.iloc[i]
            elif price_below_sma.iloc[i]:
                if current_regime['is_trending'] or not current_regime['is_high_vol']:
                    signals.iloc[i, signals.columns.get_loc('Signal')] = -1
                    signals.iloc[i, signals.columns.get_loc('Confidence')] = confidence.iloc[i]
        
        # Calculate dynamic stops and position sizes
        stops = self.risk_manager.calculate_dynamic_stops(data)
        signals['Stop_Loss_Long'] = stops['atr_stop_long']
        signals['Stop_Loss_Short'] = stops['atr_stop_short']
        
        return signals

class MultiTimeframeStrategy(Strategy):
    """
    Strategy that combines signals from multiple timeframes
    """
    
    def __init__(self, timeframes: List[str] = ['1H', '4H', '1D']):
        self.timeframes = timeframes
        self.strategies = {
            tf: AdaptiveSMAStrategy(base_window=50) for tf in timeframes
        }
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = data.copy()
        signals['Signal'] = 0
        signals['MTF_Score'] = 0
        
        # Get signals from each timeframe
        timeframe_signals = {}
        for tf in self.timeframes:
            # Resample data to timeframe
            tf_data = self._resample_data(data, tf)
            tf_signals = self.strategies[tf].generate_signals(tf_data)
            
            # Align back to original timeframe
            aligned_signals = self._align_signals(tf_signals, data.index)
            timeframe_signals[tf] = aligned_signals
        
        # Combine signals with weights (longer timeframes get more weight)
        weights = {'1H': 0.3, '4H': 0.4, '1D': 0.3}
        
        for i in range(len(signals)):
            mtf_score = 0
            for tf in self.timeframes:
                if tf in timeframe_signals and i < len(timeframe_signals[tf]):
                    tf_signal = timeframe_signals[tf]['Signal'].iloc[i] if i < len(timeframe_signals[tf]) else 0
                    tf_confidence = timeframe_signals[tf].get('Confidence', pd.Series([1.0] * len(timeframe_signals[tf]))).iloc[i] if i < len(timeframe_signals[tf]) else 1.0
                    mtf_score += tf_signal * weights.get(tf, 0.33) * tf_confidence
            
            signals.iloc[i, signals.columns.get_loc('MTF_Score')] = mtf_score
            
            # Generate final signal
            if mtf_score > 0.3:
                signals.iloc[i, signals.columns.get_loc('Signal')] = 1
            elif mtf_score < -0.3:
                signals.iloc[i, signals.columns.get_loc('Signal')] = -1
        
        return signals
    
    def _resample_data(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample data to specified timeframe"""
        ohlc_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }
        return data.resample(timeframe).agg(ohlc_dict).dropna()
    
    def _align_signals(self, tf_signals: pd.DataFrame, target_index: pd.Index) -> pd.DataFrame:
        """Align timeframe signals back to original index"""
        return tf_signals.reindex(target_index, method='ffill')

class MLEnhancedStrategy(Strategy):
    """
    Strategy enhanced with simple machine learning features
    """
    
    def __init__(self, base_strategy=None, feature_window: int = 20):
        self.base_strategy = base_strategy or AdaptiveSMAStrategy()
        self.feature_window = feature_window
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # Get base signals
        signals = self.base_strategy.generate_signals(data)
        
        # Add ML features
        features = self._calculate_features(data)
        
        # Simple ML scoring (replace with actual ML model)
        ml_score = self._calculate_ml_score(features)
        
        # Combine base signals with ML score
        signals['ML_Score'] = ml_score
        signals['Enhanced_Signal'] = 0
        
        for i in range(len(signals)):
            base_signal = signals['Signal'].iloc[i]
            ml_score_val = ml_score.iloc[i] if i < len(ml_score) else 0
            
            # Only take signals when ML agrees
            if base_signal == 1 and ml_score_val > 0.1:
                signals.iloc[i, signals.columns.get_loc('Enhanced_Signal')] = 1
            elif base_signal == -1 and ml_score_val < -0.1:
                signals.iloc[i, signals.columns.get_loc('Enhanced_Signal')] = -1
        
        # Use enhanced signals
        signals['Signal'] = signals['Enhanced_Signal']
        
        return signals
    
    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate ML features"""
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['price_sma_ratio'] = data['Close'] / data['Close'].rolling(20).mean()
        features['price_change'] = data['Close'].pct_change()
        features['price_momentum'] = data['Close'].pct_change(5)
        
        # Volume features
        features['volume_sma_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
        features['volume_price_trend'] = (data['Volume'] * data['Close'].pct_change()).rolling(5).mean()
        
        # Volatility features
        features['volatility'] = data['Close'].pct_change().rolling(20).std()
        features['price_range'] = (data['High'] - data['Low']) / data['Close']
        
        # Technical features
        rsi = self._calculate_rsi(data['Close'])
        features['rsi'] = rsi
        features['rsi_divergence'] = rsi - rsi.rolling(10).mean()
        
        return features.fillna(0)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_ml_score(self, features: pd.DataFrame) -> pd.Series:
        """Simple ML scoring (replace with trained model)"""
        # Simple linear combination (placeholder for real ML model)
        weights = {
            'price_sma_ratio': 0.3,
            'price_momentum': 0.2,
            'volume_sma_ratio': 0.1,
            'volatility': -0.1,
            'rsi_divergence': 0.2,
            'price_range': -0.1
        }
        
        score = pd.Series(0, index=features.index)
        for feature, weight in weights.items():
            if feature in features.columns:
                normalized_feature = (features[feature] - features[feature].rolling(50).mean()) / features[feature].rolling(50).std()
                score += normalized_feature.fillna(0) * weight
        
        return score

class PortfolioStrategy:
    """
    Manages multiple strategies as a portfolio
    """
    
    def __init__(self, strategies: Dict[str, Strategy], weights: Dict[str, float]):
        self.strategies = strategies
        self.weights = weights
        self.risk_manager = AdvancedRiskManager()
        
    def generate_portfolio_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate combined portfolio signals"""
        portfolio_signals = data.copy()
        portfolio_signals['Portfolio_Signal'] = 0
        portfolio_signals['Portfolio_Confidence'] = 0
        
        # Get signals from each strategy
        strategy_signals = {}
        for name, strategy in self.strategies.items():
            strategy_signals[name] = strategy.generate_signals(data)
        
        # Combine signals
        for i in range(len(portfolio_signals)):
            weighted_signal = 0
            total_confidence = 0
            
            for name, weight in self.weights.items():
                if name in strategy_signals and i < len(strategy_signals[name]):
                    signal = strategy_signals[name]['Signal'].iloc[i]
                    confidence = strategy_signals[name].get('Confidence', pd.Series([1.0] * len(strategy_signals[name]))).iloc[i] if i < len(strategy_signals[name]) else 1.0
                    
                    weighted_signal += signal * weight * confidence
                    total_confidence += confidence * weight
            
            portfolio_signals.iloc[i, portfolio_signals.columns.get_loc('Portfolio_Signal')] = np.sign(weighted_signal) if abs(weighted_signal) > 0.3 else 0
            portfolio_signals.iloc[i, portfolio_signals.columns.get_loc('Portfolio_Confidence')] = min(total_confidence, 1.0)
        
        return portfolio_signals
