# 📈 Multi-Asset Quantitative Strategy Terminal

> **🧠 Deep Dive:** Want to understand the calculus and statistics powering this engine? **[Read the full Quantitative Math Guide here](QUANT_MATH_GUIDE.md)**.

## Overview
The Multi-Asset Quantitative Strategy Terminal is an advanced, Python-based web application designed for comprehensive financial analysis, time-series forecasting, and portfolio optimization. Built with Streamlit, this tool bridges the gap between traditional technical analysis and institutional-grade quantitative modeling. It allows users to forecast stocks, currencies, cryptocurrencies, market indices, and commodities using classic statistics, machine learning, and macroeconomic drivers.


⚠️ Disclaimer: This application is for educational and research purposes only. It is not financial advice. The forecasting models and portfolio optimizations are based on historical data, and past performance does not guarantee future results. Do your own research or consult a certified financial advisor before making investment decisions. The creator of this software assumes no liability for any financial losses.

## Core Features
* **Technical Analysis Dashboard:** Interactive primary charts featuring togglable Simple Moving Averages (SMA 50 & 200) and Relative Strength Index (RSI) overlays.
* **Mathematical Grading (RMSE):** Automatically calculates the Root Mean Square Error (RMSE) for all forecasting models, allowing you to mathematically validate historical accuracy.
* **Classic Statistical Forecasting (Tab 1):** Highly tunable ARIMA, SARIMA (Seasonal), and Exponential Smoothing (ETS) models for momentum and baseline time-series forecasting.
* **Machine Learning & Algorithmic (Tab 2):** Leverages Facebook Prophet for trend identification and a Random Forest Regressor for calendar-based pattern recognition.
* **Macroeconomic SARIMAX (Tab 3):** Offers multivariate forecasting by mathematically factoring in exogenous macroeconomic variables. Secure session states allow for direct exporting of forecast data (`.csv`) and detailed statistical summaries (`.pdf`).
* **Macro Correlation Scanner (Tab 4):** Automatically generates a correlation heatmap comparing your target asset against global drivers (US Dollar, Gold, Crude Oil, 10Y Yield, S&P 500) to identify optimal exogenous variables.
* **Modern Portfolio Optimizer (Tab 5):** Utilizes `scipy` quadratic programming to calculate the Markowitz Efficient Frontier. Input multiple assets to find the exact capital allocation that maximizes the Sharpe Ratio (highest return for the lowest risk).
* **Built-in Strategy Guide (Tab 6):** An educational overview of different asset classes and instructions on how to navigate the terminal's modules.

## Technology Stack
This application relies on the following core libraries:
* **Streamlit:** For the interactive web interface and session state management.
* **yfinance:** To download live historical market data.
* **Pandas & NumPy:** For robust data manipulation, rolling calculations, and structuring.
* **Plotly:** To render interactive financial charts, heatmaps, and donut graphs.
* **Statsmodels:** For implementing ARIMA, SARIMAX, and Exponential Smoothing algorithms.
* **Prophet:** For automated forecasting of 24/7 time-series data.
* **Scikit-Learn:** For running the Random Forest regressor and calculating RMSE scores.
* **SciPy:** For running Sequential Least Squares Programming (SLSQP) in portfolio optimization.
* **FPDF:** For generating lightweight, downloadable PDF summary reports.

## Installation and Setup
**1. Clone the repository**
Download the project files to your local machine.

**2. Install dependencies**
Ensure you have Python installed, then install the required packages using the provided requirements file:
pip install -r requirements.txt

**3. Run the application**
Launch the Streamlit server from your terminal:
streamlit run app.py

## How to Use
**1. Global Configuration:** Use the sidebar to enter your target ticker symbol (e.g., AAPL, EURUSD=X, BTC-USD) and set your historical date range and forecast horizon.  <br>
**2. Technical Analysis:** Toggle the SMA and RSI checkboxes in the sidebar to read current market conditions. <br>
**3. Identify Drivers:** Navigate to Tab 4 (Macro Scanner) to find which global macroeconomic indicators are inversely or directly correlated to your asset.  <br>
**4. Forecast Future Prices:** Use Tabs 1, 2, or 3 to run statistical or machine learning models. Use the RMSE grade to determine which model is the most accurate for your specific asset.  <br>
**5. Export Data:** In Tab 3, download your raw CSV forecasts and PDF statistical summaries for external use.  <br>
**6. Optimize Capital:** Navigate to Tab 5, enter a basket of tickers, and calculate the mathematically perfect portfolio weights to maximize your risk-adjusted returns.
