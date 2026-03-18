# 📈 The Ultimate Multi-Model Market Predictor

## Overview
The Ultimate Multi-Model Market Predictor is an interactive, Python-based web application designed to forecast stocks, currencies, and commodities. Built with Streamlit, this tool allows users to pull live financial data and apply a variety of forecasting techniques—ranging from classic statistical models to modern machine learning algorithms—to predict future asset prices.

## Core Features
- **Live Data Integration:** Fetches historical market data dynamically based on user-defined tickers and date ranges.
- **Classic Statistical Modeling:** Includes customizable ARIMA, SARIMA (Seasonal), and Exponential Smoothing (ETS) models for baseline time-series forecasting.
- **Algorithmic & Machine Learning:** Leverages Facebook Prophet for trend identification and a Random Forest Regressor for pattern-based predictions.
- **Macroeconomic SARIMAX:** Offers multivariate forecasting by mathematically factoring in exogenous macroeconomic variables (like the 10-Year US Treasury Yield, US Dollar Index, or Crude Oil).
- **Built-in Market Guide:** Provides an educational overview of different asset classes (Stocks, Forex, Commodities) to help users select the most appropriate forecasting model.
- **Interactive Visualizations:** Generates dynamic, interactive charts to visualize historical data alongside predictive forecasts.

## Technology Stack
This application relies on the following core libraries:
- Streamlit: For the interactive web interface.
- yfinance: To download historical market data from Yahoo Finance.
- Pandas: For robust data manipulation and structuring.
- Plotly: To render interactive financial charts.
- Statsmodels: For implementing ARIMA, SARIMAX, and Exponential Smoothing algorithms.
- Prophet: For automated forecasting of time-series data.

## Installation and Setup
**1. Clone the repository**
Download the project files to your local machine.

**2. Install dependencies**
Ensure you have Python installed, then install the required packages using the provided requirements file:

Bash
pip install -r requirements.txt

**3. Run the application**
Launch the Streamlit server from your terminal:

Bash
streamlit run app.py


## How to Use
**1. Configure Data:** Open the sidebar to input your target ticker symbol (e.g., AAPL for Apple, EURUSD=X for Forex).

**2. Set Timeframe:** Select your historical start and end dates, and use the slider to determine how many days into the future you want to predict.

**3. Choose a Methodology:** Navigate through the application tabs to select your preferred forecasting approach (Statistical, ML & Prophet, or Macro SARIMAX).

**4. Tune Parameters:** Adjust the mathematical sliders (like p, d, q for ARIMA) to fit the model to the specific behavior of your chosen asset.

**5. Generate Forecast:** Click the run button within your chosen tab to process the data and generate your interactive prediction chart.
