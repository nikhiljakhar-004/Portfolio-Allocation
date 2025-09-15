# risk_models.py

import pandas as pd
import numpy as np

def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the daily percentage returns from a price series.
    """
    # Calculate the percentage change between the current and a prior element.
    daily_returns = prices.pct_change()
    # Drop the first row, which will be NaN (Not a Number) since it has no prior day to compare to.
    return daily_returns.dropna()

def get_expected_returns(daily_returns: pd.DataFrame) -> np.ndarray:
    """
    Calculates the annualized expected returns from daily returns.
    """
    # Calculate the mean of daily returns for each stock and annualize it.
    # We multiply by 252, the typical number of trading days in a year.
    return daily_returns.mean() * 252

def get_covariance_matrix(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the annualized covariance matrix of returns.
    """
    # Calculate the covariance of the daily returns and annualize it.
    # This matrix is fundamental for measuring portfolio risk.
    return daily_returns.cov() * 252