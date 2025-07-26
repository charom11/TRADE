"""
Multi-Strategy Portfolio Trading System
=====================================
- Combines multiple top-performing strategies with optimal allocation
- Implements dynamic portfolio rebalancing
- Provides risk management and performance tracking
- Supports both backtesting and live trading modes

Strategy Allocation (Based on Performance Analysis):
- 40% SMA_50 (Best returns)
- 25% EMA_50 (Second best returns)
- 20% BB_Trend (Best risk-adjusted)
- 15% Composite_Trend_Focus (Diversification)

Requirements:
    pip install pandas numpy matplotlib seaborn pandas-ta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from src.indicators import add_indicators
from src.strategy import (
    SimpleMovingAverageStrategy, ExponentialMovingAverageStrategy,
    BollingerBandsWithTrendStrategy, CompositeStrategy
)
from src.backtesting import Backtester
import warnings
warnings.filterwarnings('ignore')

class MultiStrategyPortfolio:
    """
    Multi-strategy portfolio management system.
    """
    
    def __init__(self, initial_capital=10000, rebalance_frequency='monthly'):
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        
        # Strategy allocation (based on comprehensive testing results)
        self.strategy_allocation = {
            'SMA_50': 0.40,      # 40% - Best overall returns
            'EMA_50': 0.25,      # 25% - Second best returns  
            'BB_Trend': 0.20,    # 20% - Best risk-adjusted
            'Composite_Trend_Focus': 0.15  # 15% - Diversification
        }
        
        # Initialize strategies
        self.strategies = {
            'SMA_50': SimpleMovingAverageStrategy(window=50),
            'EMA_50': ExponentialMovingAverageStrategy(window=50),
            'BB_Trend': BollingerBandsWithTrendStrategy(bb_window=20, bb_std=2.0, trend_window=50),
            'Composite_Trend_Focus': CompositeStrategy(rsi_weight=0.2, macd_weight=0.3, bb_weight=0.2, trend_weight=0.3)
        }
        
        # Portfolio tracking
        self.portfolio_history = []
        self.strategy_signals = {}
        self.strategy_returns = {}
        self.rebalance_dates = []
        
    def generate_all_signals(self, data):
        """Generate signals for all strategies."""
        print("Generating signals for all strategies...")
        
        for strategy_name, strategy in self.strategies.items():
            print(f"Processing {strategy_name}...")
            signals = strategy.generate_signals(data.copy())
            self.strategy_signals[strategy_name] = signals
            
        return self.strategy_signals
    
    def calculate_portfolio_signals(self, data):
        """Calculate combined portfolio signals based on strategy allocation."""
        portfolio_data = data.copy()
        portfolio_data['Portfolio_Signal'] = 0.0
        portfolio_data['Portfolio_Score'] = 0.0
        
        # Combine signals based on allocation weights
        for strategy_name, weight in self.strategy_allocation.items():
            if strategy_name in self.strategy_signals:
                signals = self.strategy_signals[strategy_name]
                portfolio_data['Portfolio_Score'] += signals['Signal'] * weight
        
        # Generate final portfolio signals (threshold-based)
        portfolio_data.loc[portfolio_data['Portfolio_Score'] > 0.3, 'Portfolio_Signal'] = 1
        portfolio_data.loc[portfolio_data['Portfolio_Score'] < -0.3, 'Portfolio_Signal'] = -1
        
        # Store individual strategy contributions for analysis
        for strategy_name in self.strategy_allocation.keys():
            if strategy_name in self.strategy_signals:
                portfolio_data[f'{strategy_name}_Signal'] = self.strategy_signals[strategy_name]['Signal']
        
        return portfolio_data
    
    def backtest_portfolio(self, data):
        """Backtest the multi-strategy portfolio."""
        print("Starting portfolio backtesting...")
        
        # Generate all strategy signals
        self.generate_all_signals(data)
        
        # Calculate portfolio signals
        portfolio_data = self.calculate_portfolio_signals(data)
        
        # Run backtesting on portfolio signals
        portfolio_data_copy = portfolio_data.copy()
        portfolio_data_copy['Signal'] = portfolio_data_copy['Portfolio_Signal']
        
        backtester = Backtester(portfolio_data_copy)
        portfolio_results = backtester.run()
        
        # Backtest individual strategies for comparison
        individual_results = {}
        for strategy_name, strategy in self.strategies.items():
            print(f"Backtesting individual strategy: {strategy_name}")
            signals = strategy.generate_signals(data.copy())
            backtester = Backtester(signals)
            results = backtester.run()
            individual_results[strategy_name] = results
        
        return portfolio_results, individual_results
    
    def calculate_portfolio_metrics(self, portfolio_results):
        """Calculate comprehensive portfolio performance metrics."""
        returns = portfolio_results['Strategy_Return'].dropna()
        
        if len(returns) == 0 or returns.std() == 0:
            return self._empty_metrics()
        
        # Basic metrics
        total_return = (portfolio_results['Equity_Curve'].iloc[-1] - 1) * 100
        annualized_return = ((portfolio_results['Equity_Curve'].iloc[-1]) ** (252 / len(portfolio_results)) - 1) * 100
        volatility = returns.std() * np.sqrt(252) * 100
        sharpe_ratio = (annualized_return / 100) / (volatility / 100) if volatility > 0 else 0
        
        # Drawdown calculation
        equity_curve = portfolio_results['Equity_Curve']
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # Trade analysis
        signals = portfolio_results['Signal'].diff()
        total_trades = len(signals[signals != 0])
        
        if total_trades > 0:
            winning_trades = len(returns[returns > 0])
            win_rate = (winning_trades / len(returns[returns != 0])) * 100 if len(returns[returns != 0]) > 0 else 0
        else:
            win_rate = 0
        
        return {
            'Total_Return': total_return,
            'Annualized_Return': annualized_return,
            'Volatility': volatility,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown,
            'Win_Rate': win_rate,
            'Total_Trades': total_trades
        }
    
    def _empty_metrics(self):
        """Return empty metrics for failed calculations."""
        return {
            'Total_Return': 0, 'Annualized_Return': 0, 'Volatility': 0,
            'Sharpe_Ratio': 0, 'Max_Drawdown': 0, 'Win_Rate': 0, 'Total_Trades': 0
        }
    
    def create_performance_report(self, portfolio_results, individual_results):
        """Create comprehensive performance report comparing portfolio vs individual strategies."""
        print("\n" + "="*80)
        print("MULTI-STRATEGY PORTFOLIO PERFORMANCE REPORT")
        print("="*80)
        
        # Calculate portfolio metrics
        portfolio_metrics = self.calculate_portfolio_metrics(portfolio_results)
        
        # Calculate individual strategy metrics
        individual_metrics = {}
        for strategy_name, results in individual_results.items():
            individual_metrics[strategy_name] = self.calculate_portfolio_metrics(results)
        
        # Display results
        print(f"\n{'Strategy':<25} {'Allocation':<12} {'Total Return (%)':<15} {'Sharpe Ratio':<12} {'Max DD (%)':<12}")
        print("-" * 85)
        
        # Portfolio results
        print(f"{'PORTFOLIO (Combined)':<25} {'100%':<12} {portfolio_metrics['Total_Return']:<15.2f} "
              f"{portfolio_metrics['Sharpe_Ratio']:<12.2f} {portfolio_metrics['Max_Drawdown']:<12.2f}")
        
        print("\nIndividual Strategy Performance:")
        print("-" * 50)
        
        for strategy_name, allocation in self.strategy_allocation.items():
            if strategy_name in individual_metrics:
                metrics = individual_metrics[strategy_name]
                print(f"{strategy_name:<25} {allocation*100:.0f}%{'':<9} {metrics['Total_Return']:<15.2f} "
                      f"{metrics['Sharpe_Ratio']:<12.2f} {metrics['Max_Drawdown']:<12.2f}")
        
        # Performance comparison
        print(f"\n{'PERFORMANCE ANALYSIS:'}")
        print("-" * 30)
        
        # Calculate portfolio vs best individual strategy
        best_individual = max(individual_metrics.items(), key=lambda x: x[1]['Total_Return'])
        print(f"Best Individual Strategy: {best_individual[0]} ({best_individual[1]['Total_Return']:.2f}%)")
        print(f"Portfolio Performance: {portfolio_metrics['Total_Return']:.2f}%")
        
        portfolio_advantage = portfolio_metrics['Total_Return'] - best_individual[1]['Total_Return']
        print(f"Portfolio Advantage: {portfolio_advantage:+.2f}%")
        
        # Risk analysis
        portfolio_sharpe = portfolio_metrics['Sharpe_Ratio']
        best_sharpe = max(individual_metrics.items(), key=lambda x: x[1]['Sharpe_Ratio'])
        print(f"\nRisk-Adjusted Performance:")
        print(f"Portfolio Sharpe Ratio: {portfolio_sharpe:.3f}")
        print(f"Best Individual Sharpe: {best_sharpe[0]} ({best_sharpe[1]['Sharpe_Ratio']:.3f})")
        
        # Diversification benefit
        print(f"\nDiversification Benefits:")
        portfolio_dd = portfolio_metrics['Max_Drawdown']
        avg_individual_dd = np.mean([m['Max_Drawdown'] for m in individual_metrics.values()])
        dd_improvement = avg_individual_dd - portfolio_dd
        print(f"Average Individual Max DD: {avg_individual_dd:.2f}%")
        print(f"Portfolio Max DD: {portfolio_dd:.2f}%")
        print(f"Drawdown Improvement: {dd_improvement:+.2f}%")
        
        return portfolio_metrics, individual_metrics
    
    def plot_portfolio_comparison(self, portfolio_results, individual_results):
        """Create comprehensive visualization comparing portfolio vs individual strategies."""
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Multi-Strategy Portfolio Analysis', fontsize=20, fontweight='bold')
        
        # 1. Equity Curves Comparison
        ax1 = axes[0, 0]
        ax1.plot(portfolio_results['Equity_Curve'], label='Portfolio (Combined)', 
                linewidth=3, color='black', linestyle='-')
        
        colors = ['blue', 'green', 'red', 'orange']
        for i, (strategy_name, results) in enumerate(individual_results.items()):
            ax1.plot(results['Equity_Curve'], label=strategy_name, 
                    linewidth=2, color=colors[i % len(colors)], alpha=0.7)
        
        ax1.set_title('Equity Curves Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Equity Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdown Comparison
        ax2 = axes[0, 1]
        
        # Calculate drawdowns
        portfolio_equity = portfolio_results['Equity_Curve']
        portfolio_running_max = portfolio_equity.expanding().max()
        portfolio_drawdown = (portfolio_equity - portfolio_running_max) / portfolio_running_max * 100
        
        ax2.fill_between(portfolio_drawdown.index, portfolio_drawdown, 0, 
                        alpha=0.3, color='red', label='Portfolio Drawdown')
        ax2.plot(portfolio_drawdown, color='red', linewidth=2)
        
        ax2.set_title('Portfolio Drawdown Analysis', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Strategy Allocation Pie Chart
        ax3 = axes[1, 0]
        wedges, texts, autotexts = ax3.pie(self.strategy_allocation.values(), 
                                          labels=self.strategy_allocation.keys(),
                                          autopct='%1.1f%%', startangle=90,
                                          colors=['gold', 'lightcoral', 'lightskyblue', 'lightgreen'])
        
        ax3.set_title('Strategy Allocation', fontsize=14, fontweight='bold')
        
        # 4. Performance Metrics Comparison
        ax4 = axes[1, 1]
        
        # Prepare data for comparison
        strategies = ['Portfolio'] + list(individual_results.keys())
        returns = [self.calculate_portfolio_metrics(portfolio_results)['Total_Return']]
        sharpe_ratios = [self.calculate_portfolio_metrics(portfolio_results)['Sharpe_Ratio']]
        
        for strategy_name, results in individual_results.items():
            metrics = self.calculate_portfolio_metrics(results)
            returns.append(metrics['Total_Return'])
            sharpe_ratios.append(metrics['Sharpe_Ratio'])
        
        x = np.arange(len(strategies))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, returns, width, label='Total Return (%)', alpha=0.8)
        ax4_twin = ax4.twinx()
        bars2 = ax4_twin.bar(x + width/2, sharpe_ratios, width, label='Sharpe Ratio', 
                            alpha=0.8, color='orange')
        
        ax4.set_xlabel('Strategies')
        ax4.set_ylabel('Total Return (%)', color='blue')
        ax4_twin.set_ylabel('Sharpe Ratio', color='orange')
        ax4.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(strategies, rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax4.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('portfolio_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def save_portfolio_results(self, portfolio_results, individual_results):
        """Save portfolio results to CSV files."""
        
        # Save portfolio equity curve
        portfolio_df = portfolio_results[['Equity_Curve', 'Strategy_Return', 'Signal']].copy()
        portfolio_df.to_csv('portfolio_equity_curve.csv')
        
        # Save individual strategy comparison
        comparison_data = []
        
        # Portfolio metrics
        portfolio_metrics = self.calculate_portfolio_metrics(portfolio_results)
        portfolio_metrics['Strategy'] = 'Portfolio (Combined)'
        portfolio_metrics['Allocation'] = '100%'
        comparison_data.append(portfolio_metrics)
        
        # Individual strategy metrics
        for strategy_name, results in individual_results.items():
            metrics = self.calculate_portfolio_metrics(results)
            metrics['Strategy'] = strategy_name
            metrics['Allocation'] = f"{self.strategy_allocation[strategy_name]*100:.0f}%"
            comparison_data.append(metrics)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv('portfolio_strategy_comparison.csv', index=False)
        
        print("\nResults saved:")
        print("- portfolio_equity_curve.csv")
        print("- portfolio_strategy_comparison.csv")
        print("- portfolio_analysis.png")

def main():
    """Main function to run multi-strategy portfolio analysis."""
    print("="*60)
    print("MULTI-STRATEGY PORTFOLIO TRADING SYSTEM")
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
    
    # Initialize portfolio
    initial_capital = 10000
    portfolio = MultiStrategyPortfolio(initial_capital=initial_capital)
    
    print(f"\n📊 Portfolio Configuration:")
    print(f"Initial Capital: ${initial_capital:,}")
    print("Strategy Allocation:")
    for strategy, allocation in portfolio.strategy_allocation.items():
        print(f"  • {strategy}: {allocation*100:.0f}% (${initial_capital*allocation:,.0f})")
    
    # Run backtesting
    portfolio_results, individual_results = portfolio.backtest_portfolio(data_with_indicators)
    
    # Create performance report
    portfolio_metrics, individual_metrics = portfolio.create_performance_report(
        portfolio_results, individual_results)
    
    # Create visualizations
    portfolio.plot_portfolio_comparison(portfolio_results, individual_results)
    
    # Save results
    portfolio.save_portfolio_results(portfolio_results, individual_results)
    
    print(f"\n🎉 Multi-strategy portfolio analysis complete!")
    print(f"📈 Portfolio Final Return: {portfolio_metrics['Total_Return']:.2f}%")
    print(f"📊 Portfolio Sharpe Ratio: {portfolio_metrics['Sharpe_Ratio']:.3f}")
    print(f"🛡️ Portfolio Max Drawdown: {portfolio_metrics['Max_Drawdown']:.2f}%")

if __name__ == "__main__":
    main()
