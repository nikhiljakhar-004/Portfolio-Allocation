# final.py

import pandas as pd
import numpy as np

# Import all the functions we've built
from fetch_data import fetch_data
from risk_models import calculate_returns, get_expected_returns, get_covariance_matrix
from portfolio_optimizer import find_optimal_risky_portfolio, calculate_efficient_frontier, portfolio_performance
from visuals import plot_results, display_portfolio_weights
# Import our new Black-Litterman functions
from black_litterman import calculate_implied_returns, black_litterman_returns

def get_user_model_choice():
    """
    Prompts the user in the terminal to choose which model to run.
    """
    while True:
        choice = input("Do you want to use the advanced Black-Litterman model? (yes/no): ").lower().strip()
        if choice in ['yes', 'y']:
            return True
        elif choice in ['no', 'n']:
            return False
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

def main():
    """
    Main function to run the entire portfolio optimization and analysis pipeline.
    """
    # --- INTERACTIVE MODEL SELECTION ---
    USE_BLACK_LITTERMAN = get_user_model_choice()

    # --- General Configuration ---
    tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS']
    start_date = '2020-01-01'
    end_date = '2024-12-31'
    risk_free_rate = 0.07

    # --- Step 1: Data Fetching & Basic Models ---
    print("\nStep 1: Fetching data and calculating basic risk models...")
    price_data = fetch_data(tickers, start_date, end_date)
    if price_data.empty:
        return
    returns_data = calculate_returns(price_data)
    cov_matrix = get_covariance_matrix(returns_data)
    print("Data and risk models ready.")

    # --- Step 2: Calculate Expected Returns (Based on the User's Choice) ---
    print("\nStep 2: Calculating Expected Returns...")
    
    if USE_BLACK_LITTERMAN:
        print("Strategy: Using Black-Litterman Model")
        market_caps = {
            'RELIANCE.NS': 1900000, 'TCS.NS': 1400000, 'HDFCBANK.NS': 1100000,
            'INFY.NS': 650000, 'ICICIBANK.NS': 780000, 'HINDUNILVR.NS': 550000
        }
        delta = 2.5
        tau = 0.05
        P = np.array([[0, 1, 0, -1, 0, 0], [1, 0, 0, 0, 0, 0]])
        Q = np.array([0.04, 0.15])

        implied_returns = calculate_implied_returns(delta, cov_matrix.values, market_caps, cov_matrix.columns)
        expected_returns = black_litterman_returns(implied_returns, tau, cov_matrix.values, P, Q)
    else:
        print("Strategy: Using Historical Average Returns")
        expected_returns = get_expected_returns(returns_data).values

    # --- Step 3: Portfolio Optimization ---
    print("\nStep 3: Optimizing portfolio...")
    optimal_weights = find_optimal_risky_portfolio(expected_returns, cov_matrix, risk_free_rate)
    opt_return, opt_volatility = portfolio_performance(optimal_weights, expected_returns, cov_matrix)
    opt_sharpe = (opt_return - risk_free_rate) / opt_volatility
    optimal_portfolio_results = (opt_volatility, opt_return, opt_sharpe, optimal_weights)
    print("Portfolio optimized.")

    # --- Step 4: Efficient Frontier Calculation ---
    print("\nStep 4: Calculating the efficient frontier for plotting...")
    efficient_frontier_results = calculate_efficient_frontier(expected_returns, cov_matrix)
    print("Efficient frontier calculated.")

    # --- Step 5: Results Display and Visualization ---
    print("\nStep 5: Displaying final results...")
    
    # --- ADDED BACK: Display the key performance metrics ---
    print("\n--- Final Portfolio Metrics ---")
    print(f"  - Expected Annual Return: {opt_return:.2%}")
    print(f"  - Annual Volatility (Risk): {opt_volatility:.2%}")
    print(f"  - Sharpe Ratio: {opt_sharpe:.2f}")
    print("---------------------------------")
    
    # Display the final weights in the console.
    display_portfolio_weights(optimal_weights, price_data.columns)
    
    # Show the final chart.
    plot_results(efficient_frontier_results, optimal_portfolio_results, risk_free_rate)


if __name__ == '__main__':
    main()