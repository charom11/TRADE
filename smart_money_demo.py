"""
Smart Money Concepts Strategy Demo
=================================
Demonstrates the Smart Money Concepts strategy based on LuxAlgo's indicator
Shows detailed analysis of:
- Market Structure Breaks (BOS/CHoCH)
- Order Blocks detection and mitigation
- Fair Value Gaps identification
- Equal Highs/Lows detection
- Premium/Discount zones analysis

This implementation respects the CC BY-NC-SA 4.0 license
Original work © LuxAlgo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.indicators import add_indicators
from src.smart_money_strategy import SmartMoneyConceptsStrategy
from src.backtesting import Backtester
import warnings
warnings.filterwarnings('ignore')

def analyze_smc_signals(data, signals):
    """Analyze Smart Money Concepts signals in detail"""
    
    print("\n" + "="*60)
    print("SMART MONEY CONCEPTS SIGNAL ANALYSIS")
    print("="*60)
    
    # Structure breaks analysis
    structure_breaks = signals[signals['Structure_Signal'] != 0]
    if len(structure_breaks) > 0:
        print(f"\n📊 MARKET STRUCTURE BREAKS: {len(structure_breaks)} total")
        print("-" * 40)
        
        bos_count = len(structure_breaks[structure_breaks['Structure_Type'] == 'BOS'])
        choch_count = len(structure_breaks[structure_breaks['Structure_Type'] == 'CHoCH'])
        bullish_breaks = len(structure_breaks[structure_breaks['Structure_Signal'] == 1])
        bearish_breaks = len(structure_breaks[structure_breaks['Structure_Signal'] == -1])
        
        print(f"Break of Structure (BOS): {bos_count}")
        print(f"Change of Character (CHoCH): {choch_count}")
        print(f"Bullish Breaks: {bullish_breaks}")
        print(f"Bearish Breaks: {bearish_breaks}")
        
        # Show recent structure breaks
        print(f"\nRecent Structure Breaks:")
        recent_breaks = structure_breaks.tail(5)[['Close', 'Structure_Type', 'Structure_Signal', 'SMC_Score']]
        for idx, row in recent_breaks.iterrows():
            direction = "BULLISH" if row['Structure_Signal'] == 1 else "BEARISH"
            print(f"  {idx.strftime('%Y-%m-%d %H:%M')} | {direction} {row['Structure_Type']} | Price: ${row['Close']:.2f} | Score: {row['SMC_Score']:.2f}")
    
    # Signal distribution analysis
    total_signals = len(signals[signals['Signal'] != 0])
    buy_signals = len(signals[signals['Signal'] == 1])
    sell_signals = len(signals[signals['Signal'] == -1])
    
    print(f"\n📈 TRADING SIGNALS: {total_signals} total")
    print("-" * 30)
    print(f"Buy Signals: {buy_signals} ({buy_signals/total_signals*100:.1f}%)")
    print(f"Sell Signals: {sell_signals} ({sell_signals/total_signals*100:.1f}%)")
    
    # SMC Score analysis
    avg_score = signals['SMC_Score'].abs().mean()
    max_score = signals['SMC_Score'].abs().max()
    
    print(f"\n🎯 SIGNAL STRENGTH:")
    print("-" * 20)
    print(f"Average SMC Score: {avg_score:.3f}")
    print(f"Maximum SMC Score: {max_score:.3f}")
    
    # Entry reasons analysis
    entry_reasons = []
    for reason_str in signals[signals['Entry_Reason'] != '']['Entry_Reason']:
        if reason_str:
            reasons = [r.strip() for r in reason_str.split(',')]
            entry_reasons.extend(reasons)
    
    if entry_reasons:
        from collections import Counter
        reason_counts = Counter(entry_reasons)
        
        print(f"\n🔍 ENTRY REASONS FREQUENCY:")
        print("-" * 30)
        for reason, count in reason_counts.most_common(10):
            print(f"{reason}: {count} times")
    
    # Zone analysis
    premium_time = len(signals[signals['Premium_Zone'] == 1])
    discount_time = len(signals[signals['Discount_Zone'] == 1])
    equilibrium_time = len(signals[signals['Equilibrium_Zone'] == 1])
    total_time = len(signals)
    
    print(f"\n🏢 MARKET ZONES DISTRIBUTION:")
    print("-" * 35)
    print(f"Premium Zone: {premium_time} bars ({premium_time/total_time*100:.1f}%)")
    print(f"Discount Zone: {discount_time} bars ({discount_time/total_time*100:.1f}%)")
    print(f"Equilibrium Zone: {equilibrium_time} bars ({equilibrium_time/total_time*100:.1f}%)")

def create_smc_visualization(data, signals):
    """Create comprehensive Smart Money Concepts visualization"""
    
    fig, axes = plt.subplots(4, 1, figsize=(20, 16))
    fig.suptitle('Smart Money Concepts Analysis Dashboard', fontsize=20, fontweight='bold')
    
    # 1. Price action with signals
    ax1 = axes[0]
    ax1.plot(data.index, data['Close'], label='Price', color='black', linewidth=1)
    
    # Mark buy/sell signals
    buy_signals = signals[signals['Signal'] == 1]
    sell_signals = signals[signals['Signal'] == -1]
    
    ax1.scatter(buy_signals.index, buy_signals['Close'], 
               color='green', marker='^', s=100, label='Buy Signal', zorder=5)
    ax1.scatter(sell_signals.index, sell_signals['Close'], 
               color='red', marker='v', s=100, label='Sell Signal', zorder=5)
    
    ax1.set_title('Price Action with SMC Signals', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. SMC Score over time
    ax2 = axes[1]
    ax2.plot(signals.index, signals['SMC_Score'], color='purple', linewidth=2, label='SMC Score')
    ax2.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='Buy Threshold')
    ax2.axhline(y=-0.5, color='red', linestyle='--', alpha=0.7, label='Sell Threshold')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.fill_between(signals.index, signals['SMC_Score'], 0, 
                     where=(signals['SMC_Score'] > 0), color='green', alpha=0.3)
    ax2.fill_between(signals.index, signals['SMC_Score'], 0, 
                     where=(signals['SMC_Score'] < 0), color='red', alpha=0.3)
    
    ax2.set_title('Smart Money Concepts Score', fontsize=14, fontweight='bold')
    ax2.set_ylabel('SMC Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Market structure breaks
    ax3 = axes[2]
    ax3.plot(data.index, data['Close'], color='black', linewidth=1, alpha=0.5)
    
    # Mark structure breaks
    structure_breaks = signals[signals['Structure_Signal'] != 0]
    bullish_breaks = structure_breaks[structure_breaks['Structure_Signal'] == 1]
    bearish_breaks = structure_breaks[structure_breaks['Structure_Signal'] == -1]
    
    ax3.scatter(bullish_breaks.index, bullish_breaks['Close'], 
               color='blue', marker='o', s=150, label='Bullish Structure Break', 
               edgecolors='white', linewidth=2, zorder=5)
    ax3.scatter(bearish_breaks.index, bearish_breaks['Close'], 
               color='orange', marker='o', s=150, label='Bearish Structure Break', 
               edgecolors='white', linewidth=2, zorder=5)
    
    ax3.set_title('Market Structure Breaks', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Price ($)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Premium/Discount zones
    ax4 = axes[3]
    ax4.plot(data.index, data['Close'], color='black', linewidth=1, label='Price')
    
    # Color background based on zones
    premium_zones = signals[signals['Premium_Zone'] == 1]
    discount_zones = signals[signals['Discount_Zone'] == 1]
    equilibrium_zones = signals[signals['Equilibrium_Zone'] == 1]
    
    for idx in premium_zones.index:
        ax4.axvspan(idx, idx, alpha=0.3, color='red', label='Premium' if idx == premium_zones.index[0] else "")
    
    for idx in discount_zones.index:
        ax4.axvspan(idx, idx, alpha=0.3, color='green', label='Discount' if idx == discount_zones.index[0] else "")
    
    for idx in equilibrium_zones.index:
        ax4.axvspan(idx, idx, alpha=0.3, color='gray', label='Equilibrium' if idx == equilibrium_zones.index[0] else "")
    
    ax4.set_title('Premium/Discount Zones', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Price ($)')
    ax4.set_xlabel('Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('smart_money_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def compare_with_traditional_strategies(data_with_indicators):
    """Compare Smart Money Concepts with traditional strategies"""
    
    print("\n" + "="*60)
    print("SMART MONEY vs TRADITIONAL STRATEGIES COMPARISON")
    print("="*60)
    
    # Import traditional strategies
    from src.strategy import SimpleMovingAverageStrategy, RSIStrategy
    
    strategies = {
        'Smart Money Concepts': SmartMoneyConceptsStrategy(swing_length=50, order_block_count=5),
        'SMA 50': SimpleMovingAverageStrategy(window=50),
        'RSI Dynamic': RSIStrategy(base_oversold=30, base_overbought=70, volatility_factor=0.5)
    }
    
    results = {}
    
    for name, strategy in strategies.items():
        print(f"\nTesting {name}...")
        try:
            signals = strategy.generate_signals(data_with_indicators)
            backtester = Backtester(signals)
            backtest_results = backtester.run()
            
            # Calculate metrics
            total_return = (backtest_results['Equity_Curve'].iloc[-1] - 1) * 100
            returns = backtest_results['Strategy_Return'].dropna()
            sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            
            # Drawdown
            equity_curve = backtest_results['Equity_Curve']
            running_max = equity_curve.expanding().max()
            drawdown = (equity_curve - running_max) / running_max * 100
            max_drawdown = drawdown.min()
            
            results[name] = {
                'Total Return (%)': total_return,
                'Sharpe Ratio': sharpe_ratio,
                'Max Drawdown (%)': max_drawdown,
                'Total Trades': len(signals[signals['Signal'] != 0])
            }
            
        except Exception as e:
            print(f"Error testing {name}: {e}")
            results[name] = {
                'Total Return (%)': 0,
                'Sharpe Ratio': 0,
                'Max Drawdown (%)': 0,
                'Total Trades': 0
            }
    
    # Display comparison
    df = pd.DataFrame(results).T
    print(f"\n{'Strategy':<25} {'Total Return (%)':<15} {'Sharpe Ratio':<12} {'Max DD (%)':<12} {'Trades':<8}")
    print("-" * 75)
    
    for strategy_name, metrics in df.iterrows():
        print(f"{strategy_name:<25} {metrics['Total Return (%)']:<15.2f} {metrics['Sharpe Ratio']:<12.3f} "
              f"{metrics['Max Drawdown (%)']:<12.2f} {metrics['Total Trades']:<8.0f}")
    
    return df

def main():
    """Main function for Smart Money Concepts demonstration"""
    
    print("="*60)
    print("SMART MONEY CONCEPTS STRATEGY DEMONSTRATION")
    print("Based on LuxAlgo's Smart Money Concepts Indicator")
    print("="*60)
    
    # Load data
    csv_path = 'BTCUSDT_binance_historical_data.csv'
    try:
        data = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
        print(f"✅ Successfully loaded data from {csv_path}")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
    
    # Resample to hourly data
    TIME_FRAME = '1H'
    ohlc_dict = {
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }
    data_resampled = data.resample(TIME_FRAME).agg(ohlc_dict).dropna()
    print(f"✅ Data resampled to {TIME_FRAME} timeframe")
    
    # Add indicators
    data_with_indicators = add_indicators(data_resampled.copy())
    print("✅ Technical indicators calculated")
    
    # Initialize Smart Money Concepts strategy
    smc_strategy = SmartMoneyConceptsStrategy(
        swing_length=50,
        internal_length=5,
        order_block_count=5,
        signal_threshold=0.5
    )
    
    print(f"\n📊 Strategy Configuration:")
    print(f"Swing Length: 50 bars")
    print(f"Internal Length: 5 bars") 
    print(f"Order Block Count: 5")
    print(f"Signal Threshold: 0.5")
    
    # Generate signals
    print("\n🔄 Generating Smart Money Concepts signals...")
    signals = smc_strategy.generate_signals(data_with_indicators)
    print("✅ Signals generated successfully")
    
    # Analyze signals
    analyze_smc_signals(data_with_indicators, signals)
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    create_smc_visualization(data_with_indicators, signals)
    
    # Run backtest
    print("\n📈 Running backtest...")
    backtester = Backtester(signals)
    backtest_results = backtester.run()
    
    # Calculate performance metrics
    total_return = (backtest_results['Equity_Curve'].iloc[-1] - 1) * 100
    returns = backtest_results['Strategy_Return'].dropna()
    sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    
    equity_curve = backtest_results['Equity_Curve']
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max * 100
    max_drawdown = drawdown.min()
    
    win_rate = len(returns[returns > 0]) / len(returns[returns != 0]) * 100 if len(returns[returns != 0]) > 0 else 0
    
    print(f"\n🎯 SMART MONEY CONCEPTS PERFORMANCE:")
    print("-" * 40)
    print(f"Total Return: {total_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Total Trades: {len(signals[signals['Signal'] != 0])}")
    
    # Compare with traditional strategies
    comparison_df = compare_with_traditional_strategies(data_with_indicators)
    
    # Save results
    signals.to_csv('smart_money_signals.csv')
    backtest_results.to_csv('smart_money_backtest.csv')
    comparison_df.to_csv('smart_money_comparison.csv')
    
    print(f"\n💾 Results saved:")
    print("- smart_money_signals.csv")
    print("- smart_money_backtest.csv") 
    print("- smart_money_comparison.csv")
    print("- smart_money_analysis.png")

if __name__ == "__main__":
    main()
