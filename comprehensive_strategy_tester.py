"""
Comprehensive Strategy Tester
----------------------------
- Tests all enhanced trading strategies on historical data
- Compares performance metrics including returns, Sharpe ratio, max drawdown
- Identifies the best performing strategy
- Generates detailed performance reports

Requirements:
    pip install pandas numpy matplotlib seaborn pandas-ta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.indicators import add_indicators
from src.strategy import (
    SimpleMovingAverageStrategy, ExponentialMovingAverageStrategy, 
    RSIStrategy, BollingerBandsStrategy, MACDStrategy,
    BollingerBandsWithTrendStrategy, CompositeStrategy,
    MeanReversionStrategy, BreakoutStrategy
)
from src.smart_money_strategy import SmartMoneyConceptsStrategy
from src.backtesting import Backtester
import warnings
warnings.filterwarnings('ignore')

def calculate_performance_metrics(results):
    """Calculate comprehensive performance metrics."""
    returns = results['Strategy_Return'].dropna()
    
    if len(returns) == 0 or returns.std() == 0:
        return {
            'Total_Return': 0,
            'Annualized_Return': 0,
            'Volatility': 0,
            'Sharpe_Ratio': 0,
            'Max_Drawdown': 0,
            'Win_Rate': 0,
            'Profit_Factor': 0,
            'Total_Trades': 0
        }
    
    # Basic metrics
    total_return = (results['Equity_Curve'].iloc[-1] - 1) * 100
    annualized_return = ((results['Equity_Curve'].iloc[-1]) ** (252 / len(results)) - 1) * 100
    volatility = returns.std() * np.sqrt(252) * 100
    sharpe_ratio = (annualized_return / 100) / (volatility / 100) if volatility > 0 else 0
    
    # Drawdown calculation
    equity_curve = results['Equity_Curve']
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max * 100
    max_drawdown = drawdown.min()
    
    # Trade analysis
    signals = results['Signal'].diff()
    total_trades = len(signals[signals != 0])
    
    if total_trades > 0:
        winning_trades = len(returns[returns > 0])
        win_rate = (winning_trades / len(returns[returns != 0])) * 100 if len(returns[returns != 0]) > 0 else 0
        
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    else:
        win_rate = 0
        profit_factor = 0
    
    return {
        'Total_Return': total_return,
        'Annualized_Return': annualized_return,
        'Volatility': volatility,
        'Sharpe_Ratio': sharpe_ratio,
        'Max_Drawdown': max_drawdown,
        'Win_Rate': win_rate,
        'Profit_Factor': profit_factor,
        'Total_Trades': total_trades
    }

def test_all_strategies():
    """Test all enhanced trading strategies."""
    print("Starting comprehensive strategy testing...")
    
    # Load data
    csv_path = 'BTCUSDT_binance_historical_data.csv'
    try:
        data = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
        print(f"Successfully loaded data from {csv_path}")
    except Exception as e:
        print(f"Failed to load data: {e}")
        return
    
    # Resample to hourly data
    TIME_FRAME = '1H'
    ohlc_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
    }
    data_resampled = data.resample(TIME_FRAME).agg(ohlc_dict).dropna()
    print(f"Data resampled to {TIME_FRAME} timeframe")
    
    # Add comprehensive indicators
    data_with_indicators = add_indicators(data_resampled.copy())
    
    # Define strategies to test
    strategies = {
        'SMA_20': SimpleMovingAverageStrategy(window=20),
        'SMA_50': SimpleMovingAverageStrategy(window=50),
        'EMA_20': ExponentialMovingAverageStrategy(window=20),
        'EMA_50': ExponentialMovingAverageStrategy(window=50),
        'RSI_Dynamic': RSIStrategy(base_oversold=30, base_overbought=70, volatility_factor=0.5),
        'RSI_Conservative': RSIStrategy(base_oversold=25, base_overbought=75, volatility_factor=0.3),
        'BB_Basic': BollingerBandsStrategy(),
        'BB_Trend': BollingerBandsWithTrendStrategy(bb_window=20, bb_std=2.0, trend_window=50),
        'MACD_Enhanced': MACDStrategy(histogram_threshold=0.1, trend_filter=True),
        'MACD_Aggressive': MACDStrategy(histogram_threshold=0.05, trend_filter=False),
        'Composite_Balanced': CompositeStrategy(rsi_weight=0.25, macd_weight=0.25, bb_weight=0.25, trend_weight=0.25),
        'Composite_Trend_Focus': CompositeStrategy(rsi_weight=0.2, macd_weight=0.3, bb_weight=0.2, trend_weight=0.3),
        'MeanReversion_Conservative': MeanReversionStrategy(lookback=20, z_threshold=2.5, rsi_period=14),
        'MeanReversion_Aggressive': MeanReversionStrategy(lookback=15, z_threshold=2.0, rsi_period=14),
        'Breakout_Standard': BreakoutStrategy(channel_period=20, volume_threshold=1.5),
        'Breakout_Sensitive': BreakoutStrategy(channel_period=15, volume_threshold=1.3),
        'SmartMoney_Standard': SmartMoneyConceptsStrategy(swing_length=50, order_block_count=5),
        'SmartMoney_Aggressive': SmartMoneyConceptsStrategy(swing_length=30, order_block_count=8),
    }
    
    results_summary = {}
    equity_curves = {}
    
    # Test each strategy
    for strategy_name, strategy in strategies.items():
        print(f"\nTesting {strategy_name}...")
        try:
            signals = strategy.generate_signals(data_with_indicators)
            backtester = Backtester(signals)
            results = backtester.run()
            
            # Calculate performance metrics
            metrics = calculate_performance_metrics(results)
            results_summary[strategy_name] = metrics
            equity_curves[strategy_name] = results['Equity_Curve']
            
            print(f"{strategy_name} - Total Return: {metrics['Total_Return']:.2f}%, Sharpe: {metrics['Sharpe_Ratio']:.2f}")
            
        except Exception as e:
            print(f"Error testing {strategy_name}: {e}")
            results_summary[strategy_name] = {
                'Total_Return': -100, 'Annualized_Return': -100, 'Volatility': 0,
                'Sharpe_Ratio': -10, 'Max_Drawdown': -100, 'Win_Rate': 0,
                'Profit_Factor': 0, 'Total_Trades': 0
            }
    
    return results_summary, equity_curves, data_with_indicators

def create_performance_report(results_summary):
    """Create and display performance report."""
    print("\n" + "="*80)
    print("COMPREHENSIVE STRATEGY PERFORMANCE REPORT")
    print("="*80)
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(results_summary).T
    df = df.sort_values('Total_Return', ascending=False)
    
    print(f"\n{'Strategy':<25} {'Total Return (%)':<15} {'Sharpe Ratio':<12} {'Max DD (%)':<12} {'Win Rate (%)':<12}")
    print("-" * 85)
    
    for strategy, metrics in df.iterrows():
        print(f"{strategy:<25} {metrics['Total_Return']:<15.2f} {metrics['Sharpe_Ratio']:<12.2f} "
              f"{metrics['Max_Drawdown']:<12.2f} {metrics['Win_Rate']:<12.2f}")
    
    # Find best strategies by different metrics
    best_return = df.loc[df['Total_Return'].idxmax()]
    best_sharpe = df.loc[df['Sharpe_Ratio'].idxmax()]
    best_drawdown = df.loc[df['Max_Drawdown'].idxmax()]  # Least negative
    
    print(f"\n{'BEST PERFORMERS:':<25}")
    print(f"{'Best Total Return:':<25} {best_return.name} ({best_return['Total_Return']:.2f}%)")
    print(f"{'Best Sharpe Ratio:':<25} {best_sharpe.name} ({best_sharpe['Sharpe_Ratio']:.2f})")
    print(f"{'Best Max Drawdown:':<25} {best_drawdown.name} ({best_drawdown['Max_Drawdown']:.2f}%)")
    
    # Overall recommendation
    df['Composite_Score'] = (
        df['Total_Return'] / df['Total_Return'].max() * 0.4 +
        df['Sharpe_Ratio'] / df['Sharpe_Ratio'].max() * 0.3 +
        (1 + df['Max_Drawdown'] / df['Max_Drawdown'].min()) * 0.3
    )
    
    best_overall = df.loc[df['Composite_Score'].idxmax()]
    print(f"{'OVERALL BEST STRATEGY:':<25} {best_overall.name}")
    print(f"{'Composite Score:':<25} {best_overall['Composite_Score']:.3f}")
    
    return df

def plot_equity_curves(equity_curves):
    """Plot equity curves for comparison."""
    plt.figure(figsize=(15, 10))
    
    # Plot top 8 strategies by final value
    final_values = {name: curve.iloc[-1] for name, curve in equity_curves.items()}
    top_strategies = sorted(final_values.items(), key=lambda x: x[1], reverse=True)[:8]
    
    for i, (strategy_name, _) in enumerate(top_strategies):
        plt.plot(equity_curves[strategy_name], label=strategy_name, linewidth=2)
    
    plt.title('Top 8 Strategy Equity Curves Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Equity Curve', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('strategy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run comprehensive strategy testing."""
    # Test all strategies
    results_summary, equity_curves, data = test_all_strategies()
    
    # Create performance report
    performance_df = create_performance_report(results_summary)
    
    # Plot equity curves
    plot_equity_curves(equity_curves)
    
    # Save detailed results
    performance_df.to_csv('strategy_performance_report.csv')
    print(f"\nDetailed performance report saved to 'strategy_performance_report.csv'")
    
    # Show correlation matrix of strategy returns
    plt.figure(figsize=(12, 8))
    
    # Calculate correlation of top strategies
    top_8_names = list(dict(sorted({name: curve.iloc[-1] for name, curve in equity_curves.items()}.items(), 
                                 key=lambda x: x[1], reverse=True)[:8]).keys())
    
    returns_matrix = pd.DataFrame({name: equity_curves[name].pct_change().fillna(0) 
                                 for name in top_8_names})
    
    correlation_matrix = returns_matrix.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'label': 'Correlation'})
    plt.title('Strategy Returns Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('strategy_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
