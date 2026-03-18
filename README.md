# 📈 The Multi-Asset Quantitative Terminal

## Overview
[span_0](start_span)The Multi-Asset Quantitative Terminal is an interactive, Python-based web application designed to forecast a wide variety of financial instruments, including stocks, currencies, cryptocurrencies, market indices, and commodities[span_0](end_span). [span_1](start_span)Built with Streamlit, this tool allows users to pull live financial data and apply a variety of forecasting techniques—ranging from classic statistical models to modern machine learning algorithms—to predict future asset prices[span_1](end_span). 

## Core Features
* **[span_2](start_span)Live Data Integration:** Fetches historical market data dynamically based on user-defined tickers and date ranges[span_2](end_span). Now supports 24/7 assets like Cryptocurrencies without breaking on weekend gaps.
* **Mathematical Grading (RMSE):** Automatically calculates the Root Mean Square Error (RMSE) for every model, allowing you to mathematically prove which algorithm performed best on historical data before trusting its future forecast.
* **Export & Reporting:** Temporarily saves session states to allow users to download raw forecast data as `.csv` files and dense statistical summaries as perfectly formatted `.pdf` reports.
* **[span_3](start_span)Classic Statistical Modeling:** Includes customizable ARIMA, SARIMA (Seasonal), and Exponential Smoothing (ETS) models for baseline time-series forecasting[span_3](end_span).
* **[span_4](start_span)Algorithmic & Machine Learning:** Leverages Facebook Prophet for trend identification and a Random Forest Regressor for pattern-based predictions[span_4](end_span).
* **[span_5](start_span)Macroeconomic SARIMAX:** Offers multivariate forecasting by mathematically factoring in exogenous macroeconomic variables (like the 10-Year US Treasury Yield, US Dollar Index, or Crude Oil)[span_5](end_span).
* **[span_6](start_span)Built-in Asset Class Guide:** Provides an educational overview of different asset classes (Stocks, Forex, Crypto, Real Estate, Commodities) to help users select the most appropriate forecasting model[span_6](end_span).
* **[span_7](start_span)Interactive Visualizations:** Generates dynamic, interactive charts to visualize historical data alongside predictive forecasts[span_7](end_span).

## Technology Stack
This application relies on the following core libraries:
* **[span_8](start_span)Streamlit:** For the interactive web interface[span_8](end_span).
* **[span_9](start_span)yfinance:** To download historical market data from Yahoo Finance[span_9](end_span).
* **[span_10](start_span)Pandas & NumPy:** For robust data manipulation and structuring[span_10](end_span).
* **[span_11](start_span)Plotly:** To render interactive financial charts[span_11](end_span).
* **[span_12](start_span)Statsmodels:** For implementing ARIMA, SARIMAX, and Exponential Smoothing algorithms[span_12](end_span).
* **[span_13](start_span)Prophet:** For automated forecasting of time-series data[span_13](end_span).
* **Scikit-Learn:** For calculating the Root Mean Square Error (RMSE) and running the Random Forest model.
* **FPDF:** For generating lightweight, downloadable PDF summary reports.

## Installation and Setup
**1. Clone the repository**
[span_14](start_span)Download the project files to your local machine[span_14](end_span).

**2. Install dependencies**
[span_15](start_span)Ensure you have Python installed, then install the required packages using the provided requirements file[span_15](end_span):
```bash
pip install -r requirements.txt
