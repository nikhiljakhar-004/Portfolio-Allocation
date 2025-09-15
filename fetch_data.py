# fetch_data.py

import pandas as pd
import yfinance as yf
from typing import List

def fetch_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical adjusted closing prices for a list of tickers from Yahoo Finance.
    """
    try:
        # Download the data from Yahoo Finance
        data = yf.download(tickers, start=start_date, end=end_date)['Close']

        # Clean the data by removing any assets that have no data at all
        data.dropna(axis=1, how='all', inplace=True)

        # Fill any remaining missing values
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)

        return data
    except Exception as e:
        print(f"An error occurred while fetching data: {e}")
        return pd.DataFrame()