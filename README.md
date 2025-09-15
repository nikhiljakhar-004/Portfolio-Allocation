# Advanced Portfolio Allocation & Optimization

This project implements a sophisticated portfolio allocation tool in Python. It constructs an optimal investment portfolio by leveraging two distinct models for calculating expected returns: a simple model based on historical averages and an advanced model using the Black-Litterman formula. The application is interactive, allowing the user to choose which model to use for the final optimization.

## Overview

The primary goal of this project is to build an efficient investment portfolio that maximizes returns for a given level of risk (i.e., maximizes the Sharpe Ratio). It achieves this by moving beyond simple historical models, which can be unstable and unreliable. The core of this project is the implementation of the Black-Litterman model, which creates stable, forward-looking return estimates by blending market equilibrium returns with an investor's specific views.

The project is structured in a modular way, with each Python script handling a specific part of the workflow:
-   **`fetch_data.py`**: Retrieves historical stock price data.
-   **`risk_models.py`**: Calculates historical returns and the covariance matrix (the risk model).
-   **`black_litterman.py`**: Implements the advanced Black-Litterman model for expected returns.
-   **`portfolio_optimizer.py`**: Contains the core optimization engine to find the best portfolio     weights.
-   **`visuals.py`**: Handles the presentation of results, including plotting and weight display.
-   **`final.py`**: The main script that orchestrates the entire process from data fetching to final output.

## Features

-   **Dual Model Approach:** Allows the user to choose between using simple historical returns or the advanced Black-Litterman model for optimization.
-   **Data Fetching:** Downloads historical adjusted closing prices for a given list of tickers from Yahoo Finance.
-   **Risk Modeling:** Calculates the annualized covariance matrix, a fundamental input for measuring portfolio risk.
-   **Advanced Return Estimates:** Implements the Black-Litterman model to generate sophisticated, blended expected returns.
-   **Mean-Variance Optimization:** Uses the `scipy.optimize` library to find the optimal risky portfolio that maximizes the Sharpe Ratio, subject to diversification constraints.
-   **Efficient Frontier Calculation:** Computes and visualizes the efficient frontier, showing the set of optimal portfolios.
-   **Rich Visualization:** Plots the efficient frontier, the Capital Market Line (CML), and the optimal portfolio in a clear and professional chart.

## Technologies Used

-   **Language:** Python
-   **Libraries:**
    -   pandas
    -   numpy
    -   scipy
    -   yfinance
    -   matplotlib
    -   seaborn
