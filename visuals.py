# visuals.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_results(efficient_frontier_results, optimal_risky_portfolio_results, risk_free_rate):
    """
    Plots the efficient frontier, the optimal risky portfolio, and the Capital Market Line (CML).
    """
    ef_volatility, ef_returns = efficient_frontier_results
    opt_vol, opt_ret, opt_sharpe, _ = optimal_risky_portfolio_results

    # Use a professional-looking plot style.
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the efficient frontier curve.
    ax.plot(ef_volatility, ef_returns, 'g--', linewidth=2.0, label='Efficient Frontier')
    
    # Plot the Optimal Risky Portfolio as a large red star.
    ax.scatter(opt_vol, opt_ret, marker='*', color='r', s=250, label=f'Optimal Risky Portfolio (Sharpe: {opt_sharpe:.2f})')
    
    # Plot the Capital Market Line (CML).
    cml_x = [0, opt_vol, opt_vol * 1.5]
    cml_y = [risk_free_rate, opt_ret, risk_free_rate + opt_sharpe * (opt_vol * 1.5)]
    ax.plot(cml_x, cml_y, color='b', linestyle='-', linewidth=1.5, label='Capital Market Line (CML)')

    # Set titles and labels for clarity.
    ax.set_title('Efficient Frontier & Capital Market Line', fontsize=18)
    ax.set_xlabel('Annualized Volatility (Risk)', fontsize=14)
    ax.set_ylabel('Annualized Return', fontsize=14)
    ax.legend(loc='best', fontsize=12)
    
    plt.tight_layout()
    # Display the plot in a new window.
    plt.show()

def display_portfolio_weights(weights, tickers):
    """
    Displays the weights of the optimal portfolio in a clean table format.
    """
    weights_df = pd.DataFrame(weights, index=tickers, columns=['Weight'])
    weights_df.index.name = 'Ticker'
    print("\n--- Optimal Portfolio Weights ---")
    # Format the weights as percentages for easy reading.
    print(weights_df.to_string(formatters={'Weight': '{:,.2%}'.format}))