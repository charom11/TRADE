"""
Enhanced Strategy Testing Suite
==============================
Tests the new advanced strategies with risk management and regime detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from src.indicators import add_indicators
from src.strategy import SimpleMovingAverageStrategy, BollingerBandsStrategy
from src.enhanced_strategies import (
    AdaptiveSMAStrategy, 
    MultiTimeframeStrategy, 
    MLEnhancedStrategy,
    PortfolioStrategy
)
from src.risk_management import AdvancedRiskManager, MarketRegimeDetector, PortfolioHeatManager
from src.backtesting import Backtester

def calculate_enhanced_metrics(results):
    """Calculate enhanced performance metrics"""
    returns = results['Strategy_Return'].dropna()
    
    if len(returns) == 0 or returns.std() == 0:
        return {
            'Total_Return': 0,
            'Annualized_Return': 0,
            'Volatility': 0,
            'Sharpe_Ratio': 0,
            'Sortino_Ratio': 0,
            'Max_Drawdown': 0,
            'Calmar_Ratio': 0,
            'Win_Rate': 0,
            'Profit_Factor': 0,
            'Total_Trades': 0,
            'Average_Trade': 0,
            'Max_Consecutive_Wins': 0,
            'Max_Consecutive_Losses': 0
        }
    
    # Basic metrics
    total_return = (results['Equity_Curve'].iloc[-1] - 1) * 100
    annualized_return = ((results['Equity_Curve'].iloc[-1]) ** (252 / len(results)) - 1) * 100
    volatility = returns.std() * np.sqrt(252) * 100
    
    # Risk-adjusted metrics
    sharpe_ratio = (annualized_return / 100) / (volatility / 100) if volatility > 0 else 0
    
    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(252) * 100 if len(downside_returns) > 0 else 0.01
    sortino_ratio = (annualized_return / 100) / (downside_volatility / 100) if downside_volatility > 0 else 0
    
    # Drawdown analysis
    equity_curve = results['Equity_Curve']
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max * 100
    max_drawdown = abs(drawdown.min())
    
    # Calmar ratio
    calmar_ratio = (annualized_return / 100) / (max_drawdown / 100) if max_drawdown > 0 else 0
    
    # Trade analysis
    signals = results['Signal'].diff()
    trades = signals[signals != 0]
    total_trades = len(trades)
    
    if total_trades > 0:
        winning_trades = len(returns[returns > 0])
        losing_trades = len(returns[returns < 0])
        win_rate = (winning_trades / len(returns[returns != 0])) * 100 if len(returns[returns != 0]) > 0 else 0
        
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        average_trade = returns.mean() * 100
        
        # Consecutive wins/losses
        win_streak = 0
        loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        for ret in returns:
            if ret > 0:
                win_streak += 1
                loss_streak = 0
                max_win_streak = max(max_win_streak, win_streak)
            elif ret < 0:
                loss_streak += 1
                win_streak = 0
                max_loss_streak = max(max_loss_streak, loss_streak)
    else:
        win_rate = 0
        profit_factor = 0
        average_trade = 0
        max_win_streak = 0
        max_loss_streak = 0
    
    return {
        'Total_Return': total_return,
        'Annualized_Return': annualized_return,
        'Volatility': volatility,
        'Sharpe_Ratio': sharpe_ratio,
        'Sortino_Ratio': sortino_ratio,
        'Max_Drawdown': max_drawdown,
        'Calmar_Ratio': calmar_ratio,
        'Win_Rate': win_rate,
        'Profit_Factor': profit_factor,
        'Total_Trades': total_trades,
        'Average_Trade': average_trade,
        'Max_Consecutive_Wins': max_win_streak,
        'Max_Consecutive_Losses': max_loss_streak
    }

def test_enhanced_strategies():
    """Test all enhanced strategies"""
    print("🚀 Starting Enhanced Strategy Testing Suite...")
    print("=" * 60)
    
    # Load data
    csv_path = 'BTCUSDT_binance_historical_data.csv'
    try:
        data = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
        print(f"✅ Successfully loaded data from {csv_path}")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
    
    # Resample to hourly data
    TIME_FRAME = '1h'  # Fixed deprecated warning
    ohlc_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
    }
    data_resampled = data.resample(TIME_FRAME).agg(ohlc_dict).dropna()
    print(f"📊 Data resampled to {TIME_FRAME} timeframe: {len(data_resampled)} bars")
    
    # Add comprehensive indicators
    data_with_indicators = add_indicators(data_resampled.copy())
    print(f"📈 Added {len(data_with_indicators.columns) - 5} technical indicators")
    
    # Initialize advanced components
    risk_manager = AdvancedRiskManager()
    regime_detector = MarketRegimeDetector()
    portfolio_heat_manager = PortfolioHeatManager()
    
    # Test market regime detection
    print("\n🔍 Market Regime Analysis:")
    regime = regime_detector.detect_regime(data_with_indicators.tail(1000))
    print(f"   Current Regime: {'Trending' if regime['is_trending'] else 'Ranging'}")
    print(f"   Volatility: {'High' if regime['is_high_vol'] else 'Normal'}")
    print(f"   Trend Strength: {regime['trend_strength']:.4f}")
    print(f"   Range Position: {regime['range_position']:.2f}")
    
    # Define enhanced strategies to test
    strategies = {
        # Baseline for comparison
        'SMA_50_Baseline': SimpleMovingAverageStrategy(window=50),
        'BB_Baseline': BollingerBandsStrategy(),
        
        # Enhanced strategies
        'Adaptive_SMA': AdaptiveSMAStrategy(base_window=50),
        'MultiTimeframe': MultiTimeframeStrategy(timeframes=['1h', '4h']),
        'ML_Enhanced_SMA': MLEnhancedStrategy(
            base_strategy=AdaptiveSMAStrategy(base_window=50)
        ),
        
        # Portfolio combinations
        'Portfolio_Balanced': PortfolioStrategy(
            strategies={
                'adaptive_sma': AdaptiveSMAStrategy(base_window=50),
                'bb_trend': BollingerBandsStrategy()
            },
            weights={'adaptive_sma': 0.6, 'bb_trend': 0.4}
        )
    }
    
    results_summary = {}
    equity_curves = {}
    
    # Test each strategy
    for strategy_name, strategy in strategies.items():
        print(f"\n🧪 Testing {strategy_name}...")
        try:
            # Generate signals
            if hasattr(strategy, 'generate_portfolio_signals'):
                # Portfolio strategy
                signals = strategy.generate_portfolio_signals(data_with_indicators)
                signals['Signal'] = signals['Portfolio_Signal']
            else:
                signals = strategy.generate_signals(data_with_indicators)
            
            # Run backtest
            backtester = Backtester(signals)
            results = backtester.run()
            
            # Calculate enhanced metrics
            metrics = calculate_enhanced_metrics(results)
            results_summary[strategy_name] = metrics
            equity_curves[strategy_name] = results['Equity_Curve']
            
            # Print key metrics
            print(f"   📊 Total Return: {metrics['Total_Return']:.2f}%")
            print(f"   📈 Sharpe Ratio: {metrics['Sharpe_Ratio']:.3f}")
            print(f"   📉 Max Drawdown: {metrics['Max_Drawdown']:.2f}%")
            print(f"   🎯 Win Rate: {metrics['Win_Rate']:.1f}%")
            print(f"   🔄 Total Trades: {metrics['Total_Trades']}")
            
        except Exception as e:
            print(f"   ❌ Error testing {strategy_name}: {e}")
            results_summary[strategy_name] = {
                'Total_Return': -100, 'Sharpe_Ratio': -10, 'Max_Drawdown': -100,
                'Win_Rate': 0, 'Total_Trades': 0, 'Sortino_Ratio': -10, 'Calmar_Ratio': -10
            }
    
    return results_summary, equity_curves, data_with_indicators

def create_enhanced_report(results_summary):
    """Create enhanced performance report"""
    print("\n" + "=" * 80)
    print("🏆 ENHANCED STRATEGY PERFORMANCE REPORT")
    print("=" * 80)
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(results_summary).T
    df = df.sort_values('Total_Return', ascending=False)
    
    # Display comprehensive results
    print(f"\n{'Strategy':<20} {'Return%':<8} {'Sharpe':<7} {'Sortino':<8} {'Calmar':<7} {'MaxDD%':<8} {'WinRate%':<9} {'Trades':<7}")
    print("-" * 95)
    
    for strategy, metrics in df.iterrows():
        print(f"{strategy:<20} {metrics['Total_Return']:<8.1f} {metrics['Sharpe_Ratio']:<7.3f} "
              f"{metrics['Sortino_Ratio']:<8.3f} {metrics['Calmar_Ratio']:<7.3f} {metrics['Max_Drawdown']:<8.1f} "
              f"{metrics['Win_Rate']:<9.1f} {metrics['Total_Trades']:<7.0f}")
    
    # Find best performers by different metrics
    best_return = df.loc[df['Total_Return'].idxmax()]
    best_sharpe = df.loc[df['Sharpe_Ratio'].idxmax()]
    best_sortino = df.loc[df['Sortino_Ratio'].idxmax()]
    best_calmar = df.loc[df['Calmar_Ratio'].idxmax()]
    best_drawdown = df.loc[df['Max_Drawdown'].idxmin()]  # Smallest drawdown
    
    print(f"\n🎖️  CHAMPIONS:")
    print(f"   🥇 Best Total Return:    {best_return.name} ({best_return['Total_Return']:.2f}%)")
    print(f"   📈 Best Sharpe Ratio:    {best_sharpe.name} ({best_sharpe['Sharpe_Ratio']:.3f})")
    print(f"   🛡️  Best Sortino Ratio:   {best_sortino.name} ({best_sortino['Sortino_Ratio']:.3f})")
    print(f"   ⚖️  Best Calmar Ratio:    {best_calmar.name} ({best_calmar['Calmar_Ratio']:.3f})")
    print(f"   🔒 Best Max Drawdown:    {best_drawdown.name} ({best_drawdown['Max_Drawdown']:.2f}%)")
    
    # Calculate composite score with enhanced weighting
    df['Enhanced_Score'] = (
        (df['Total_Return'] / df['Total_Return'].max()) * 0.25 +
        (df['Sharpe_Ratio'] / df['Sharpe_Ratio'].max()) * 0.20 +
        (df['Sortino_Ratio'] / df['Sortino_Ratio'].max()) * 0.20 +
        (df['Calmar_Ratio'] / df['Calmar_Ratio'].max()) * 0.15 +
        (1 - df['Max_Drawdown'] / df['Max_Drawdown'].max()) * 0.20
    )
    
    best_overall = df.loc[df['Enhanced_Score'].idxmax()]
    print(f"\n🏆 OVERALL CHAMPION: {best_overall.name}")
    print(f"   Enhanced Score: {best_overall['Enhanced_Score']:.3f}")
    
    return df

def plot_enhanced_comparison(equity_curves, results_summary):
    """Create enhanced visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Plot 1: Equity Curves (Top performers)
    df_summary = pd.DataFrame(results_summary).T
    top_strategies = df_summary.nlargest(5, 'Total_Return').index
    
    for strategy in top_strategies:
        if strategy in equity_curves:
            ax1.plot(equity_curves[strategy], label=strategy, linewidth=2)
    
    ax1.set_title('🏆 Top 5 Strategy Equity Curves', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Equity Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Risk-Return Scatter
    returns = [results_summary[s]['Total_Return'] for s in results_summary]
    sharpes = [results_summary[s]['Sharpe_Ratio'] for s in results_summary]
    
    ax2.scatter(returns, sharpes, s=100, alpha=0.7)
    for i, strategy in enumerate(results_summary.keys()):
        ax2.annotate(strategy.replace('_', '\n'), (returns[i], sharpes[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    ax2.set_title('📊 Risk-Return Profile', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Total Return (%)')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Drawdown Comparison
    drawdowns = [results_summary[s]['Max_Drawdown'] for s in results_summary]
    strategy_names = list(results_summary.keys())
    
    bars = ax3.bar(range(len(strategy_names)), drawdowns, alpha=0.7, 
                   color=['red' if dd > 20 else 'orange' if dd > 10 else 'green' for dd in drawdowns])
    ax3.set_title('📉 Maximum Drawdown Comparison', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Strategy')
    ax3.set_ylabel('Max Drawdown (%)')
    ax3.set_xticks(range(len(strategy_names)))
    ax3.set_xticklabels(strategy_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Win Rate vs Profit Factor
    win_rates = [results_summary[s]['Win_Rate'] for s in results_summary]
    profit_factors = [results_summary[s]['Profit_Factor'] for s in results_summary]
    
    ax4.scatter(win_rates, profit_factors, s=100, alpha=0.7)
    for i, strategy in enumerate(results_summary.keys()):
        ax4.annotate(strategy.replace('_', '\n'), (win_rates[i], profit_factors[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    ax4.set_title('🎯 Win Rate vs Profit Factor', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Win Rate (%)')
    ax4.set_ylabel('Profit Factor')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('enhanced_strategy_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run enhanced strategy testing"""
    # Test enhanced strategies
    results_summary, equity_curves, data = test_enhanced_strategies()
    
    # Create enhanced performance report
    performance_df = create_enhanced_report(results_summary)
    
    # Create visualizations
    plot_enhanced_comparison(equity_curves, results_summary)
    
    # Save detailed results
    performance_df.to_csv('enhanced_strategy_performance.csv')
    print(f"\n💾 Enhanced performance report saved to 'enhanced_strategy_performance.csv'")
    
    # Risk analysis
    print(f"\n🛡️  RISK ANALYSIS:")
    low_risk_strategies = performance_df[performance_df['Max_Drawdown'] < 20]
    high_return_strategies = performance_df[performance_df['Total_Return'] > 50]
    
    if len(low_risk_strategies) > 0:
        print(f"   Low Risk Strategies (<20% DD): {', '.join(low_risk_strategies.index)}")
    if len(high_return_strategies) > 0:
        print(f"   High Return Strategies (>50%): {', '.join(high_return_strategies.index)}")
    
    # Final recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    best_balanced = performance_df.loc[performance_df['Enhanced_Score'].idxmax()]
    print(f"   🏆 Best Overall: {best_balanced.name}")
    print(f"   📊 For Conservative: Use strategy with lowest Max Drawdown")
    print(f"   🚀 For Aggressive: Use strategy with highest Total Return")
    print(f"   ⚖️  For Balanced: Use {best_balanced.name} (Enhanced Score: {best_balanced['Enhanced_Score']:.3f})")

if __name__ == "__main__":
    main()
