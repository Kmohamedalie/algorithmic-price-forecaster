import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import date, timedelta
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
import re

# --- UI SETUP ---
st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Multi-Model Stock Price Predictor")
st.markdown("Compare Statistical, Machine Learning, and Prophet models for time-series forecasting.")
st.markdown("**Disclaimer:** This application is for educational purposes only. Do not use these predictions for actual financial trading.")

# --- SIDEBAR ---
st.sidebar.header("Configuration")

# Fix 1: Robust ticker extraction. Grabs only the first ticker if the user types multiple.
raw_ticker_input = st.sidebar.text_input("Stock Ticker (e.g., AAPL, TSLA)", "AAPL").upper()
ticker = re.split(r'[,\s]+', raw_ticker_input)[0].strip()

model_choice = st.sidebar.selectbox("Select Prediction Model", 
                                    ["Facebook Prophet", "Statistical (ARIMA)", "Machine Learning (Random Forest)"])
days_to_predict = st.sidebar.slider("Days to Forecast", 1, 365, 30)

# --- DATA FETCHING ---
@st.cache_data
def load_data(ticker):
    START = "2020-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")
    
    # Download data
    data = yf.download(ticker, start=START, end=TODAY)
    
    # Fix 2: Flatten the MultiIndex columns if yfinance returns them
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading market data...')
df = load_data(ticker)
data_load_state.text(f'Data for {ticker} loaded successfully!')

st.subheader(f"Raw Data for {ticker}")
st.write(df.tail())

# --- PLOTTING RAW DATA ---
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Stock Close"))
    fig.layout.update(title_text='Historical Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

plot_raw_data()

# --- PREDICTION MODELS ---
st.subheader(f"Forecasting with {model_choice}")

if model_choice == "Facebook Prophet":
    # Prophet requires columns to be named 'ds' (date) and 'y' (value)
    df_train = df[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
    
    m = Prophet(daily_seasonality=True)
    m.fit(df_train)
    future = m.make_future_dataframe(periods=days_to_predict)
    forecast = m.predict(future)
    
    st.write("Forecast Dataframe:")
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'], name="Actual", line=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Predicted", line=dict(color='orange')))
    fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
    fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', name='Confidence Interval', fillcolor='rgba(255, 165, 0, 0.2)'))
    fig1.layout.update(title_text=f'Prophet Forecast for {ticker}')
    st.plotly_chart(fig1, use_container_width=True)

elif model_choice == "Statistical (ARIMA)":
    # ARIMA setup (Note: Hardcoded order (5,1,0) for demo)
    df_train = df['Close'].dropna().values
    
    try:
        model = ARIMA(df_train, order=(5, 1, 0))
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=days_to_predict)
        
        # Create future dates
        last_date = df['Date'].iloc[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Actual", line=dict(color='blue')))
        fig2.add_trace(go.Scatter(x=future_dates, y=forecast, name="Predicted (ARIMA)", line=dict(color='red')))
        fig2.layout.update(title_text=f'ARIMA Forecast for {ticker}')
        st.plotly_chart(fig2, use_container_width=True)
        
    except Exception as e:
        st.error(f"ARIMA model failed to converge. Try a different stock or timeframe. Error: {e}")

elif model_choice == "Machine Learning (Random Forest)":
    # Simple feature engineering: using previous days to predict the next
    df_rf = df[['Date', 'Close']].copy()
    df_rf.dropna(subset=['Close'], inplace=True)
    df_rf['Day'] = df_rf['Date'].dt.day
    df_rf['Month'] = df_rf['Date'].dt.month
    df_rf['Year'] = df_rf['Date'].dt.year
    df_rf['DayOfWeek'] = df_rf['Date'].dt.dayofweek
    
    X = df_rf[['Day', 'Month', 'Year', 'DayOfWeek']]
    y = df_rf['Close']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Predict future dates
    last_date = df['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
    future_df = pd.DataFrame({'Date': future_dates})
    future_df['Day'] = future_df['Date'].dt.day
    future_df['Month'] = future_df['Date'].dt.month
    future_df['Year'] = future_df['Date'].dt.year
    future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek
    
    forecast = model.predict(future_df[['Day', 'Month', 'Year', 'DayOfWeek']])
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Actual", line=dict(color='blue')))
    fig3.add_trace(go.Scatter(x=future_dates, y=forecast, name="Predicted (Random Forest)", line=dict(color='green')))
    fig3.layout.update(title_text=f'Random Forest Forecast for {ticker}')
    st.plotly_chart(fig3, use_container_width=True)
    
    st.warning("Note: This Random Forest uses date-derived features for simplicity. For production time-series ML, autoregressive features (lagging 'Close' prices) are highly recommended.")
