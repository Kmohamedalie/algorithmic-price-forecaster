# 📈 Algorithmic Price Forecaster

An interactive web application built with **Streamlit** that lets users forecast stock prices using three distinct algorithmic approaches — all powered by live market data fetched from Yahoo Finance via `yfinance`.

> ⚠️ **Disclaimer:** This application is built **strictly for educational and data-science demonstration purposes.** Financial markets are highly volatile, and no algorithm can predict stock prices with 100% certainty. **Do not use these models for real-world financial trading.**

---

## Features

- 🔍 **Any stock ticker** — enter any valid Yahoo Finance symbol (e.g. `AAPL`, `MSFT`, `TSLA`, `BTC-USD`)
- 📅 **Configurable date range** — choose your own historical training window
- 📆 **Adjustable forecast horizon** — predict 5 to 90 trading days into the future
- 📊 **Three forecasting models** run side-by-side with interactive Plotly charts
- 📋 **Raw data table** with one-click CSV export
- 📉 **All-model comparison chart** to visually contrast each algorithm's outlook
- 💡 **Summary metrics** — latest close, 52-week high/low, and 1-day price change

---

## Forecasting Models

### 1. 📊 ARIMA — Statistical Time-Series Analysis
**AutoRegressive Integrated Moving Average (ARIMA)** models a time series as a linear combination of its own past values and past forecast errors. This app uses an **ARIMA(5, 1, 0)** specification — 5 autoregressive lags with first-order differencing to achieve stationarity.

### 2. 🌲 Random Forest Regressor — Machine Learning
A **Random Forest Regressor** is trained on lag features derived from the closing-price series (20 lagged closing prices). Predictions are generated **iteratively**: each newly predicted value is appended to the feature window to produce the next step, enabling multi-step forecasting without retraining.

### 3. 🔮 Facebook Prophet — Additive Time-Series Modeling
**Prophet** decomposes the time series into trend, weekly seasonality, and yearly seasonality components using an additive model. It is robust to missing data and outliers and automatically handles holiday effects.

---

## Installation

**Requirements:** Python 3.9+

```bash
# 1. Clone the repository
git clone https://github.com/Kmohamedalie/algorithmic-price-forecaster.git
cd algorithmic-price-forecaster

# 2. (Recommended) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Interactive web UI |
| `yfinance` | Live stock data from Yahoo Finance |
| `pandas` / `numpy` | Data manipulation |
| `plotly` | Interactive charts |
| `statsmodels` | ARIMA model |
| `scikit-learn` | Random Forest Regressor |
| `prophet` | Facebook Prophet model |

---

## Running the App

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

### Usage

1. Enter a **stock ticker symbol** in the sidebar (default: `AAPL`)
2. Select a **start date** and **end date** for the historical training window
3. Use the **forecast horizon slider** to choose how many trading days to predict
4. Click **🚀 Run Forecast** to fetch data and train all three models
5. Explore results across the **ARIMA**, **Random Forest**, and **Prophet** tabs
6. View the **all-model comparison chart** and download the forecast as a **CSV**

---

## Project Structure

```
algorithmic-price-forecaster/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md
```

---

## License

This project is open-source and available for educational use.
