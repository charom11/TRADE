"""
Binance Data Extractor
---------------------
- Extracts historical OHLCV data from Binance using python-binance.
- Supports both interactive mode and parameter mode (edit variables at the top).
- Saves data as a CSV file.

Requirements:
    pip install python-binance pandas

Usage:
    # Parameter mode (edit variables at the top)
    python binance_data_extractor.py

    # Interactive mode
    python binance_data_extractor.py --interactive
"""

import pandas as pd
from binance.client import Client
import argparse
import sys
from datetime import datetime

# --------- Parameter Mode (edit these) ---------
SYMBOL = 'BTCUSDT'         # e.g., 'BTCUSDT', 'ETHUSDT'
INTERVAL = Client.KLINE_INTERVAL_1HOUR  # e.g., Client.KLINE_INTERVAL_1MINUTE, _1HOUR, _1DAY
START_DATE = '2018-01-01' # Format: 'YYYY-MM-DD'
END_DATE = '2025-07-24'   # End at 2025-07-24
OUTPUT_CSV = 'BTCUSDT_binance_historical_data.csv'

# --------- Interactive Mode ---------
def get_user_input():
    symbol = input('Enter symbol (e.g., BTCUSDT): ').strip().upper()
    print('Intervals: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M')
    interval = input('Enter interval (e.g., 1h): ').strip().lower()
    interval_map = {
        '1m': Client.KLINE_INTERVAL_1MINUTE,
        '3m': Client.KLINE_INTERVAL_3MINUTE,
        '5m': Client.KLINE_INTERVAL_5MINUTE,
        '15m': Client.KLINE_INTERVAL_15MINUTE,
        '30m': Client.KLINE_INTERVAL_30MINUTE,
        '1h': Client.KLINE_INTERVAL_1HOUR,
        '2h': Client.KLINE_INTERVAL_2HOUR,
        '4h': Client.KLINE_INTERVAL_4HOUR,
        '6h': Client.KLINE_INTERVAL_6HOUR,
        '8h': Client.KLINE_INTERVAL_8HOUR,
        '12h': Client.KLINE_INTERVAL_12HOUR,
        '1d': Client.KLINE_INTERVAL_1DAY,
        '3d': Client.KLINE_INTERVAL_3DAY,
        '1w': Client.KLINE_INTERVAL_1WEEK,
        '1m': Client.KLINE_INTERVAL_1MONTH,
    }
    interval = interval_map.get(interval, Client.KLINE_INTERVAL_1HOUR)
    start_date = input('Enter start date (YYYY-MM-DD): ').strip()
    end_date = input('Enter end date (YYYY-MM-DD, leave blank for up to now): ').strip()
    output_csv = input('Enter output CSV filename: ').strip()
    if not end_date:
        end_date = None
    return symbol, interval, start_date, end_date, output_csv


def fetch_binance_ohlcv(symbol, interval, start_str, end_str):
    client = Client()
    print(f"Fetching {symbol} {interval} data from {start_str} to {end_str or 'now'}...")
    try:
        klines = client.get_historical_klines(symbol, interval, start_str, end_str)
    except Exception as e:
        print(f"Error fetching data: {e}")
        sys.exit(1)
    if not klines:
        print("No data returned. Check symbol, interval, or date range.")
        sys.exit(1)
    print(f"Fetched {len(klines)} rows.")
    df = pd.DataFrame(klines, columns=[
        'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close_time', 'Quote_asset_volume', 'Number_of_trades',
        'Taker_buy_base', 'Taker_buy_quote', 'Ignore'])
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    df.set_index('Date', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    return df


def main():
    parser = argparse.ArgumentParser(description='Binance Data Extractor')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    args = parser.parse_args()

    if args.interactive:
        symbol, interval, start_date, end_date, output_csv = get_user_input()
    else:
        symbol, interval, start_date, end_date, output_csv = (
            SYMBOL, INTERVAL, START_DATE, END_DATE, OUTPUT_CSV)
        if not end_date:
            end_date = None

    df = fetch_binance_ohlcv(symbol, interval, start_date, end_date)
    df.to_csv(output_csv)
    print(f"Saved {len(df)} rows to {output_csv}")
    print(df.head())
    print(df.tail())

if __name__ == '__main__':
    main() 