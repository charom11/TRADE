import yfinance as yf
import pandas as pd
import os

def download_data(ticker, start_date, end_date, data_dir='data'):
    """
    Downloads historical stock data from Yahoo Finance and saves it to a CSV file.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL').
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.
        data_dir (str): The directory to save the data in.

    Returns:
        pd.DataFrame: A DataFrame containing the historical data.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    file_path = os.path.join(data_dir, f'{ticker}_{start_date}_to_{end_date}.csv')

    if os.path.exists(file_path):
        print(f"Data for {ticker} already exists. Loading from file.")
        data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    else:
        try:
            print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                print(f"No data found for {ticker} for the specified date range.")
                return None

            # The index from yfinance is a DatetimeIndex. We convert it to date objects.
            data.index = data.index.date
            data.index.name = 'Date'

            data.to_csv(file_path)
            print(f"Data saved to {file_path}")
        except Exception as e:
            print(f"An error occurred while downloading data for {ticker}: {e}")
            return None

    return data
