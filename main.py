from src.indicators import add_indicators
from src.strategy import SimpleMovingAverageStrategy, ExponentialMovingAverageStrategy, RSIStrategy, BollingerBandsStrategy, MACDStrategy
from src.backtesting import Backtester
from src.plotting import plot_signals
import pandas as pd

def main():
    """Main function to run and compare multiple trading strategies using local BTCUSDT data with parameter tuning for SMA and EMA windows, plus Bollinger Bands and MACD strategies."""
    print("Starting the trading bot...")

    # Configurable time frame (e.g., '1H' for hourly, '1D' for daily)
    TIME_FRAME = '1H'  # Change this to '1D', '4H', etc. as needed

    # Load data from local CSV
    csv_path = 'E:/TRADE/BTCUSDT_binance_historical_data.csv'
    ticker = 'BTCUSDT'
    try:
        data = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
        print(f"Successfully loaded data from {csv_path}.")
    except Exception as e:
        print(f"Failed to load data from {csv_path}: {e}")
        return

    # Resample data to the desired time frame
    ohlc_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
    }
    data_resampled = data.resample(TIME_FRAME).agg(ohlc_dict).dropna()
    print(f"Data resampled to {TIME_FRAME} time frame. Sample:")
    print(data_resampled.head())
        
        # Add technical indicators
    data_with_indicators = add_indicators(data_resampled.copy())

    # Parameter tuning for SMA and EMA
    windows = [9, 50, 100, 200]
    for window in windows:
        print(f"\nRunning SMA strategy with window={window}...")
        sma_strategy = SimpleMovingAverageStrategy(window=window)
        signals = sma_strategy.generate_signals(data_with_indicators)
        backtester = Backtester(signals)
        results = backtester.run()
        plot_signals(results, f"{ticker} - SMA({window}) {TIME_FRAME}")
        print(f"SMA({window}) strategy equity curve:")
        print(results[['Equity_Curve']].tail())

    for window in windows:
        print(f"\nRunning EMA strategy with window={window}...")
        ema_strategy = ExponentialMovingAverageStrategy(window=window)
        signals = ema_strategy.generate_signals(data_with_indicators)
        backtester = Backtester(signals)
        results = backtester.run()
        plot_signals(results, f"{ticker} - EMA({window}) {TIME_FRAME}")
        print(f"EMA({window}) strategy equity curve:")
        print(results[['Equity_Curve']].tail())

    # RSI strategy (unchanged)
    print(f"\nRunning RSI strategy...")
    rsi_strategy = RSIStrategy()
    signals = rsi_strategy.generate_signals(data_with_indicators)
    backtester = Backtester(signals)
    results = backtester.run()
    plot_signals(results, f"{ticker} - RSI {TIME_FRAME}")
    print(f"RSI strategy equity curve:")
    print(results[['Equity_Curve']].tail())

    # Bollinger Bands strategy
    print(f"\nRunning Bollinger Bands strategy...")
    bb_strategy = BollingerBandsStrategy()
    signals = bb_strategy.generate_signals(data_with_indicators)
    backtester = Backtester(signals)
    results = backtester.run()
    plot_signals(results, f"{ticker} - Bollinger Bands {TIME_FRAME}")
    print(f"Bollinger Bands strategy equity curve:")
    print(results[['Equity_Curve']].tail())

    # MACD strategy
    print(f"\nRunning MACD strategy...")
    macd_strategy = MACDStrategy()
    signals = macd_strategy.generate_signals(data_with_indicators)
    backtester = Backtester(signals)
    results = backtester.run()
    plot_signals(results, f"{ticker} - MACD {TIME_FRAME}")
    print(f"MACD strategy equity curve:")
    print(results[['Equity_Curve']].tail())

if __name__ == "__main__":
    main()
