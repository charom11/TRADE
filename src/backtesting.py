import pandas as pd

class Backtester:
    """
    Simple backtesting engine for long/short strategies.
    """
    def __init__(self, data: pd.DataFrame, signal_col: str = 'Signal'):
        self.data = data.copy()
        self.signal_col = signal_col

    def run(self) -> pd.DataFrame:
        self.data['Position'] = self.data[self.signal_col].shift(1).fillna(0)
        self.data['Market_Return'] = self.data['Close'].pct_change().fillna(0)
        self.data['Strategy_Return'] = self.data['Market_Return'] * self.data['Position']
        self.data['Equity_Curve'] = (1 + self.data['Strategy_Return']).cumprod()
        return self.data

