# portfolio_optimizer.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize

def portfolio_performance(weights: np.ndarray, expected_returns: np.ndarray, cov_matrix: pd.DataFrame):
    """
    Calculates the annualized portfolio performance (return and volatility).
    """
    port_return = np.sum(weights * expected_returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_return, port_volatility

def negative_sharpe_ratio(weights: np.ndarray, expected_returns: np.ndarray, cov_matrix: pd.DataFrame, risk_free_rate: float):
    """
    Calculates the negative Sharpe ratio. We minimize this function to maximize the Sharpe ratio.
    """
    p_return, p_volatility = portfolio_performance(weights, expected_returns, cov_matrix)
    return -(p_return - risk_free_rate) / p_volatility

def find_optimal_risky_portfolio(expected_returns: np.ndarray, cov_matrix: pd.DataFrame, risk_free_rate: float):
    """
    Finds the portfolio with the maximum Sharpe ratio using an optimizer,
    subject to both minimum and maximum weight constraints for each asset.
    """
    num_assets = len(expected_returns)
    args = (expected_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # --- Set minimum and maximum weight constraints ---
    min_weight = 0.05  # 5% minimum allocation
    max_weight = 0.25  # 25% maximum allocation
    
    # Safety checks for feasibility
    if min_weight * num_assets > 1:
        raise ValueError("The sum of minimum weights exceeds 100%.")
    if max_weight * num_assets < 1:
        raise ValueError("The sum of maximum weights is less than 100%, making it impossible to allocate fully.")
        
    bounds = tuple((min_weight, max_weight) for asset in range(num_assets))
    
    initial_guess = num_assets * [1. / num_assets,]
    
    result = minimize(negative_sharpe_ratio, initial_guess, args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x

def calculate_efficient_frontier(expected_returns: np.ndarray, cov_matrix: pd.DataFrame, num_portfolios: int = 100):
    """
    Calculates the efficient frontier, subject to both minimum and maximum weight constraints.
    """
    num_assets = len(expected_returns)
    args = (expected_returns, cov_matrix)
    
    # --- Set the same constraints for consistency ---
    min_weight = 0.05  # 5% minimum allocation
    max_weight = 0.25  # 25% maximum allocation
    bounds = tuple((min_weight, max_weight) for asset in range(num_assets))

    results_volatility = []
    results_returns = []
    
    target_returns = np.linspace(expected_returns.min(), expected_returns.max(), num_portfolios)
    
    for target_return in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: portfolio_performance(x, *args)[0] - target_return}
        )
        
        initial_guess = num_assets * [1. / num_assets,]
        
        result = minimize(lambda w, *args: portfolio_performance(w, *args)[1], initial_guess, args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            results_volatility.append(result.fun)
            results_returns.append(target_return)
            
    return np.array(results_volatility), np.array(results_returns)