import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from datetime import date, timedelta
from statsmodels.tsa.arima.model import ARIMA
import re
import warnings

# Suppress statsmodels warnings for cleaner terminal output
warnings.filterwarnings("ignore")

# --- UI SETUP ---
st.set_page_config(page_title="ARIMA Stock Forecaster", layout="wide")
st.title("📈 Statistical Stock Forecaster (ARIMA)")
st.markdown("""
This application uses the AutoRegressive Integrated Moving Average (ARIMA) model to forecast stock prices. 
Use the sidebar to tune the **p**, **d**, and **q** parameters and see how the forecast changes.
* **Disclaimer:** This app is for educational purposes only.
""")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("1. Data Configuration")
raw_ticker_input = st.sidebar.text_input("Stock Ticker (e.g., AAPL, TSLA)", "AAPL").upper()
ticker = re.split(r'[,\s]+', raw_ticker_input)[0].strip()
days_to_predict = st.sidebar.slider("Days to Forecast", 1, 90, 30)

st.sidebar.header("2. ARIMA Parameters")
st.sidebar.markdown("""
* **p (AutoRegressive):** Lags of the stationarized series.
* **d (Integrated):** Order of differencing.
* **q (Moving Average):** Lags of the forecast errors.
""")
p = st.sidebar.number_input("p (AR terms)", min_value=0, max_value=10, value=5)
d = st.sidebar.number_input("d (Differencing)", min_value=0, max_value=3, value=1)
q = st.sidebar.number_input("q (MA terms)", min_value=0, max_value=10, value=0)

# --- DATA FETCHING ---
@st.cache_data
def load_data(ticker):
    START = "2020-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")
    
    data = yf.download(ticker, start=START, end=TODAY)
    
    # Flatten MultiIndex if necessary
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading market data...')
df = load_data(ticker)
data_load_state.text(f'Data for {ticker} loaded successfully!')

# --- ARIMA MODELING ---
st.subheader(f"ARIMA({p},{d},{q}) Forecast for {ticker}")

# Prepare data: Drop NaNs just in case
df_train = df['Close'].dropna().values

try:
    with st.spinner('Fitting ARIMA model... This may take a moment depending on parameters.'):
        # Fit the model
        model = ARIMA(df_train, order=(p, d, q))
        fitted_model = model.fit()
        
        # Forecast
        forecast = fitted_model.forecast(steps=days_to_predict)
        
        # Generate future dates
        last_date = df['Date'].iloc[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
        
        # --- PLOTTING ---
        fig = go.Figure()
        
        # Historical Data Trace
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Close'], 
            name="Historical Actuals", 
            line=dict(color='#1f77b4')
        ))
        
        # Forecast Data Trace
        fig.add_trace(go.Scatter(
            x=future_dates, y=forecast, 
            name=f"ARIMA({p},{d},{q}) Forecast", 
            line=dict(color='#ff7f0e', width=3, dash='dot')
        ))
        
        fig.layout.update(
            title_text=f'{ticker} Price Forecast',
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            xaxis_rangeslider_visible=True,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show model summary
        with st.expander("Show ARIMA Model Statistical Summary"):
            st.text(fitted_model.summary().as_text())

except Exception as e:
    st.error(f"⚠️ The ARIMA model failed to converge with parameters ({p},{d},{q}). Try lowering the values, or check if the stock ticker is valid. Error details: {e}")
