# 📈 The Multi-Asset Quantitative Terminal

## Overview
The Multi-Asset Quantitative Terminal is an interactive, Python-based web application designed to forecast a wide variety of financial instruments, including stocks, currencies, cryptocurrencies, market indices, and commodities. Built with Streamlit, this tool allows users to pull live financial data and apply a variety of forecasting techniques—ranging from classic statistical models to modern machine learning algorithms—to predict future asset prices. 

## Core Features
* **Live Data Integration:** Fetches historical market data dynamically based on user-defined tickers and date ranges. Now supports 24/7 assets like Cryptocurrencies without breaking on weekend gaps.
* **Mathematical Grading (RMSE):** Automatically calculates the Root Mean Square Error (RMSE) for every model, allowing you to mathematically prove which algorithm performed best on historical data before trusting its future forecast.
* **Export & Reporting:** Temporarily saves session states to allow users to download raw forecast data as `.csv` files and dense statistical summaries as perfectly formatted `.pdf` reports.
* **Classic Statistical Modeling:** Includes customizable ARIMA, SARIMA (Seasonal), and Exponential Smoothing (ETS) models for baseline time-series forecasting.
* **Algorithmic & Machine Learning:** Leverages Facebook Prophet for trend identification and a Random Forest Regressor for pattern-based predictions.
* **Macroeconomic SARIMAX:** Offers multivariate forecasting by mathematically factoring in exogenous macroeconomic variables (like the 10-Year US Treasury Yield, US Dollar Index, or Crude Oil).
* **Built-in Asset Class Guide:** Provides an educational overview of different asset classes (Stocks, Forex, Crypto, Real Estate, Commodities) to help users select the most appropriate forecasting model.
* **Interactive Visualizations:** Generates dynamic, interactive charts to visualize historical data alongside predictive forecasts.

## Technology Stack
This application relies on the following core libraries:
* **Streamlit:** For the interactive web interface.
* **yfinance:** To download historical market data from Yahoo Finance.
* **Pandas & NumPy:** For robust data manipulation and structuring.
* **Plotly:** To render interactive financial charts.
* **Statsmodels:** For implementing ARIMA, SARIMAX, and Exponential Smoothing algorithms.
* **Prophet:** For automated forecasting of time-series data.
* **Scikit-Learn:** For calculating the Root Mean Square Error (RMSE) and running the Random Forest model.
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
**1. Configure Data:** Open the sidebar to input your target ticker symbol (e.g., AAPL for Apple, EURUSD=X for Forex, BTC-USD for Bitcoin).
**2. Set Timeframe:** Select your historical start and end dates, and use the slider to determine how many days into the future you want to predict.
**3. Choose a Methodology:** Navigate through the application tabs to select your preferred forecasting approach (Statistical, ML & Prophet, or Macro SARIMAX).
**4. Tune Parameters:** Adjust the mathematical sliders (like p, d, q for ARIMA) to fit the model to the specific behavior of your chosen asset. Lower your RMSE score to find the perfect fit!
**5. Generate & Export:** Click the run button within your chosen tab to process the data and generate your interactive prediction chart. Use the download buttons below the chart to export your findings.
