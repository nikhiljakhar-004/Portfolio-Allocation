# black_litterman.py

import numpy as np

def calculate_implied_returns(delta: float, sigma: np.ndarray, market_caps: dict, tickers: list) -> np.ndarray:
    """
    Calculates the implied equilibrium returns (Π) from the market portfolio.
    Formula: Π = δ * Σ * w_mkt
    """
    # Create a sorted list of market caps based on the order of tickers
    sorted_caps = np.array([market_caps[ticker] for ticker in tickers])
    
    # Calculate market-capitalization weights
    market_weights = sorted_caps / np.sum(sorted_caps)
    
    # Calculate the implied returns vector (Π)
    implied_returns = delta * sigma.dot(market_weights)
    
    return implied_returns

def black_litterman_returns(implied_returns: np.ndarray, tau: float, sigma: np.ndarray, P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Calculates the final, blended expected returns vector based on the Black-Litterman formula.
    """
    # Calculate Omega (Ω), the uncertainty matrix of the views.
    omega = np.diag(np.diag(P.dot(tau * sigma).dot(P.T)))

    # The Black-Litterman master formula:
    # E[R] = [ (τΣ)^-1 + P^T Ω^-1 P ]^-1 * [ (τΣ)^-1 Π + P^T Ω^-1 Q ]
    
    ts_inv = np.linalg.inv(tau * sigma)
    term1 = np.linalg.inv(ts_inv + P.T.dot(np.linalg.inv(omega)).dot(P))
    term2 = ts_inv.dot(implied_returns) + P.T.dot(np.linalg.inv(omega)).dot(Q)
    blended_returns = term1.dot(term2)
    
    return blended_returns