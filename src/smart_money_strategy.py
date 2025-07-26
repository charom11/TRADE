"""
Smart Money Concepts Strategy Implementation
==========================================
Based on LuxAlgo's Smart Money Concepts indicator
Implements key concepts like:
- Market Structure (BOS/CHoCH)
- Order Blocks
- Fair Value Gaps (FVG)
- Equal Highs/Lows (EQH/EQL)
- Premium/Discount Zones

This implementation respects the CC BY-NC-SA 4.0 license
Original work © LuxAlgo
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class Bias(Enum):
    BULLISH = 1
    BEARISH = -1
    NEUTRAL = 0

class StructureType(Enum):
    BOS = "BOS"  # Break of Structure
    CHOCH = "CHoCH"  # Change of Character

@dataclass
class OrderBlock:
    """Represents an order block"""
    high: float
    low: float
    time: int
    bias: Bias
    mitigated: bool = False

@dataclass
class FairValueGap:
    """Represents a Fair Value Gap"""
    top: float
    bottom: float
    time: int
    bias: Bias
    filled: bool = False

@dataclass
class SwingPoint:
    """Represents a swing high/low point"""
    price: float
    time: int
    index: int
    is_high: bool
    crossed: bool = False

class SmartMoneyStrategy:
    """
    Smart Money Concepts trading strategy
    
    This strategy implements key smart money concepts:
    - Identifies market structure breaks (BOS/CHoCH)
    - Detects order blocks for entry points
    - Finds fair value gaps for potential reversals
    - Tracks premium/discount zones for trend analysis
    """
    
    def __init__(self, 
                 swing_length: int = 50,
                 internal_length: int = 5,
                 order_block_count: int = 5,
                 fvg_enabled: bool = True,
                 equal_hl_threshold: float = 0.1):
        
        self.swing_length = swing_length
        self.internal_length = internal_length
        self.order_block_count = order_block_count
        self.fvg_enabled = fvg_enabled
        self.equal_hl_threshold = equal_hl_threshold
        
        # State variables
        self.swing_high = SwingPoint(0, 0, 0, True)
        self.swing_low = SwingPoint(0, 0, 0, False)
        self.internal_high = SwingPoint(0, 0, 0, True)
        self.internal_low = SwingPoint(0, 0, 0, False)
        
        self.swing_trend = Bias.NEUTRAL
        self.internal_trend = Bias.NEUTRAL
        
        self.order_blocks: List[OrderBlock] = []
        self.fair_value_gaps: List[FairValueGap] = []
        
        # Premium/Discount zones
        self.current_high = 0.0
        self.current_low = float('inf')
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    def detect_swing_points(self, data: pd.DataFrame, length: int = 50) -> Tuple[pd.Series, pd.Series]:
        """Detect swing highs and lows"""
        swing_highs = pd.Series(index=data.index, dtype=float)
        swing_lows = pd.Series(index=data.index, dtype=float)
        
        for i in range(length, len(data) - length):
            # Check for swing high
            current_high = data['High'].iloc[i]
            if all(current_high >= data['High'].iloc[i-length:i]) and \
               all(current_high >= data['High'].iloc[i+1:i+length+1]):
                swing_highs.iloc[i] = current_high
            
            # Check for swing low
            current_low = data['Low'].iloc[i]
            if all(current_low <= data['Low'].iloc[i-length:i]) and \
               all(current_low <= data['Low'].iloc[i+1:i+length+1]):
                swing_lows.iloc[i] = current_low
        
        return swing_highs.dropna(), swing_lows.dropna()
    
    def detect_market_structure(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect Break of Structure (BOS) and Change of Character (CHoCH)"""
        signals = data.copy()
        signals['Structure_Signal'] = 0
        signals['Structure_Type'] = ''
        signals['Structure_Bias'] = 0
        
        # Get swing points
        swing_highs, swing_lows = self.detect_swing_points(data, self.swing_length)
        
        for i in range(len(data)):
            current_close = data['Close'].iloc[i]
            
            # Check for bullish structure break
            if len(swing_highs) > 0:
                last_swing_high = swing_highs.iloc[-1] if len(swing_highs) > 0 else 0
                
                if current_close > last_swing_high and not self.swing_high.crossed:
                    structure_type = StructureType.CHOCH if self.swing_trend == Bias.BEARISH else StructureType.BOS
                    
                    signals.loc[signals.index[i], 'Structure_Signal'] = 1
                    signals.loc[signals.index[i], 'Structure_Type'] = structure_type.value
                    signals.loc[signals.index[i], 'Structure_Bias'] = Bias.BULLISH.value
                    
                    self.swing_trend = Bias.BULLISH
                    self.swing_high.crossed = True
            
            # Check for bearish structure break
            if len(swing_lows) > 0:
                last_swing_low = swing_lows.iloc[-1] if len(swing_lows) > 0 else float('inf')
                
                if current_close < last_swing_low and not self.swing_low.crossed:
                    structure_type = StructureType.CHOCH if self.swing_trend == Bias.BULLISH else StructureType.BOS
                    
                    signals.loc[signals.index[i], 'Structure_Signal'] = -1
                    signals.loc[signals.index[i], 'Structure_Type'] = structure_type.value
                    signals.loc[signals.index[i], 'Structure_Bias'] = Bias.BEARISH.value
                    
                    self.swing_trend = Bias.BEARISH
                    self.swing_low.crossed = True
        
        return signals
    
    def detect_order_blocks(self, data: pd.DataFrame) -> List[OrderBlock]:
        """Detect order blocks based on market structure"""
        order_blocks = []
        swing_highs, swing_lows = self.detect_swing_points(data, self.swing_length)
        
        # Bullish order blocks (formed before bullish structure break)
        for i, (time, high_price) in enumerate(swing_highs.items()):
            # Look for the candle that created the most bearish momentum before the break
            start_idx = data.index.get_loc(time) - self.swing_length
            end_idx = data.index.get_loc(time)
            
            if start_idx >= 0:
                segment = data.iloc[start_idx:end_idx]
                if len(segment) > 0:
                    # Find the last bearish candle before the break
                    bearish_candles = segment[segment['Close'] < segment['Open']]
                    if len(bearish_candles) > 0:
                        last_bearish = bearish_candles.iloc[-1]
                        order_block = OrderBlock(
                            high=last_bearish['High'],
                            low=last_bearish['Low'],
                            time=int(last_bearish.name.timestamp()),
                            bias=Bias.BULLISH
                        )
                        order_blocks.append(order_block)
        
        # Bearish order blocks (formed before bearish structure break)
        for i, (time, low_price) in enumerate(swing_lows.items()):
            start_idx = data.index.get_loc(time) - self.swing_length
            end_idx = data.index.get_loc(time)
            
            if start_idx >= 0:
                segment = data.iloc[start_idx:end_idx]
                if len(segment) > 0:
                    # Find the last bullish candle before the break
                    bullish_candles = segment[segment['Close'] > segment['Open']]
                    if len(bullish_candles) > 0:
                        last_bullish = bullish_candles.iloc[-1]
                        order_block = OrderBlock(
                            high=last_bullish['High'],
                            low=last_bullish['Low'],
                            time=int(last_bullish.name.timestamp()),
                            bias=Bias.BEARISH
                        )
                        order_blocks.append(order_block)
        
        return order_blocks[-self.order_block_count:] if order_blocks else []
    
    def detect_fair_value_gaps(self, data: pd.DataFrame) -> List[FairValueGap]:
        """Detect Fair Value Gaps (imbalances in price)"""
        if not self.fvg_enabled:
            return []
        
        fair_value_gaps = []
        
        for i in range(2, len(data)):
            current = data.iloc[i]
            previous = data.iloc[i-1]
            before_previous = data.iloc[i-2]
            
            # Bullish FVG: current low > before_previous high
            if current['Low'] > before_previous['High']:
                fvg = FairValueGap(
                    top=current['Low'],
                    bottom=before_previous['High'],
                    time=int(current.name.timestamp()),
                    bias=Bias.BULLISH
                )
                fair_value_gaps.append(fvg)
            
            # Bearish FVG: current high < before_previous low
            elif current['High'] < before_previous['Low']:
                fvg = FairValueGap(
                    top=before_previous['Low'],
                    bottom=current['High'],
                    time=int(current.name.timestamp()),
                    bias=Bias.BEARISH
                )
                fair_value_gaps.append(fvg)
        
        return fair_value_gaps
    
    def detect_equal_highs_lows(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect Equal Highs and Equal Lows"""
        signals = data.copy()
        signals['EQH'] = False
        signals['EQL'] = False
        
        atr = self.calculate_atr(data)
        swing_highs, swing_lows = self.detect_swing_points(data, self.swing_length)
        
        # Check for equal highs
        for i in range(1, len(swing_highs)):
            current_high = swing_highs.iloc[i]
            previous_high = swing_highs.iloc[i-1]
            current_atr = atr.loc[swing_highs.index[i]]
            
            if abs(current_high - previous_high) < self.equal_hl_threshold * current_atr:
                signals.loc[swing_highs.index[i], 'EQH'] = True
        
        # Check for equal lows
        for i in range(1, len(swing_lows)):
            current_low = swing_lows.iloc[i]
            previous_low = swing_lows.iloc[i-1]
            current_atr = atr.loc[swing_lows.index[i]]
            
            if abs(current_low - previous_low) < self.equal_hl_threshold * current_atr:
                signals.loc[swing_lows.index[i], 'EQL'] = True
        
        return signals
    
    def calculate_premium_discount_zones(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Premium and Discount zones"""
        signals = data.copy()
        
        # Update current high and low
        for i in range(len(data)):
            self.current_high = max(self.current_high, data['High'].iloc[i])
            self.current_low = min(self.current_low, data['Low'].iloc[i])
        
        # Calculate zones
        range_size = self.current_high - self.current_low
        
        # Premium zone: 70-100% of range
        premium_bottom = self.current_low + 0.7 * range_size
        
        # Discount zone: 0-30% of range  
        discount_top = self.current_low + 0.3 * range_size
        
        # Equilibrium: 45-55% of range
        equilibrium_bottom = self.current_low + 0.45 * range_size
        equilibrium_top = self.current_low + 0.55 * range_size
        
        signals['Premium_Zone'] = (data['Close'] >= premium_bottom).astype(int)
        signals['Discount_Zone'] = (data['Close'] <= discount_top).astype(int)
        signals['Equilibrium_Zone'] = ((data['Close'] >= equilibrium_bottom) & 
                                     (data['Close'] <= equilibrium_top)).astype(int)
        
        return signals
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive Smart Money Concepts signals"""
        signals = data.copy()
        signals['Signal'] = 0
        signals['SMC_Score'] = 0.0
        signals['Entry_Reason'] = ''
        
        # Detect market structure
        structure_signals = self.detect_market_structure(data)
        signals = signals.join(structure_signals[['Structure_Signal', 'Structure_Type', 'Structure_Bias']], 
                              how='left', rsuffix='_struct')
        
        # Detect order blocks
        self.order_blocks = self.detect_order_blocks(data)
        
        # Detect fair value gaps
        self.fair_value_gaps = self.detect_fair_value_gaps(data)
        
        # Detect equal highs/lows
        eqhl_signals = self.detect_equal_highs_lows(data)
        signals = signals.join(eqhl_signals[['EQH', 'EQL']], how='left', rsuffix='_eqhl')
        
        # Calculate premium/discount zones
        zone_signals = self.calculate_premium_discount_zones(data)
        signals = signals.join(zone_signals[['Premium_Zone', 'Discount_Zone', 'Equilibrium_Zone']], 
                              how='left', rsuffix='_zone')
        
        # Generate trading signals based on Smart Money Concepts
        for i in range(len(signals)):
            row = signals.iloc[i]
            score = 0.0
            reasons = []
            
            # Structure break signals
            if pd.notna(row.get('Structure_Signal', 0)) and row.get('Structure_Signal', 0) != 0:
                if row['Structure_Signal'] == 1:  # Bullish structure break
                    score += 0.4
                    reasons.append(f"Bullish {row.get('Structure_Type', 'BOS')}")
                elif row['Structure_Signal'] == -1:  # Bearish structure break
                    score -= 0.4
                    reasons.append(f"Bearish {row.get('Structure_Type', 'BOS')}")
            
            # Order block signals
            current_price = row['Close']
            for ob in self.order_blocks:
                if not ob.mitigated:
                    if ob.bias == Bias.BULLISH and ob.low <= current_price <= ob.high:
                        score += 0.3
                        reasons.append("Bullish OB")
                        ob.mitigated = True
                    elif ob.bias == Bias.BEARISH and ob.low <= current_price <= ob.high:
                        score -= 0.3
                        reasons.append("Bearish OB")
                        ob.mitigated = True
            
            # Fair Value Gap signals
            for fvg in self.fair_value_gaps:
                if not fvg.filled:
                    if fvg.bias == Bias.BULLISH and fvg.bottom <= current_price <= fvg.top:
                        score += 0.2
                        reasons.append("Bullish FVG")
                        fvg.filled = True
                    elif fvg.bias == Bias.BEARISH and fvg.bottom <= current_price <= fvg.top:
                        score -= 0.2
                        reasons.append("Bearish FVG")
                        fvg.filled = True
            
            # Zone-based signals
            if row.get('Discount_Zone', 0) == 1 and self.swing_trend == Bias.BULLISH:
                score += 0.15
                reasons.append("Discount Zone Buy")
            elif row.get('Premium_Zone', 0) == 1 and self.swing_trend == Bias.BEARISH:
                score -= 0.15
                reasons.append("Premium Zone Sell")
            
            # Equal highs/lows signals
            if row.get('EQH', False):
                score -= 0.1
                reasons.append("Equal Highs")
            elif row.get('EQL', False):
                score += 0.1
                reasons.append("Equal Lows")
            
            # Final signal generation
            signals.iloc[i, signals.columns.get_loc('SMC_Score')] = score
            signals.iloc[i, signals.columns.get_loc('Entry_Reason')] = ', '.join(reasons)
            
            if score > 0.5:
                signals.iloc[i, signals.columns.get_loc('Signal')] = 1
            elif score < -0.5:
                signals.iloc[i, signals.columns.get_loc('Signal')] = -1
        
        return signals

class SmartMoneyConceptsStrategy:
    """
    Wrapper class to integrate Smart Money Concepts with existing strategy framework
    """
    
    def __init__(self, 
                 swing_length: int = 50,
                 internal_length: int = 5,
                 order_block_count: int = 5,
                 signal_threshold: float = 0.5):
        
        self.smc = SmartMoneyStrategy(
            swing_length=swing_length,
            internal_length=internal_length,
            order_block_count=order_block_count
        )
        self.signal_threshold = signal_threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals using Smart Money Concepts"""
        return self.smc.generate_signals(data)

# Usage example and strategy registration
def create_smart_money_strategy():
    """Factory function to create Smart Money Concepts strategy"""
    return SmartMoneyConceptsStrategy(
        swing_length=50,
        internal_length=5,
        order_block_count=5,
        signal_threshold=0.5
    )
