import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from datetime import date, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import re
import warnings

# Ignore statistical convergence warnings in the UI
warnings.filterwarnings("ignore")

# --- UI SETUP ---
st.set_page_config(page_title="Statistical Stock Predictor", layout="wide")
st.title("📈 Advanced Statistical Stock Forecasting")
st.markdown("Compare industry-standard statistical models: ARIMA, SARIMA, and Exponential Smoothing.")

# --- SIDEBAR ---
st.sidebar.header("1. Data Configuration")

raw_ticker_input = st.sidebar.text_input("Stock Ticker (e.g., AAPL)", "AAPL").upper()
ticker = re.split(r'[,\s]+', raw_ticker_input)[0].strip()

start_date = st.sidebar.date_input("Start Date", date.today() - timedelta(days=365*2))
end_date = st.sidebar.date_input("End Date", date.today())

st.sidebar.header("2. Model Selection")
model_type = st.sidebar.selectbox("Choose a Statistical Model", 
                                  ["ARIMA", "SARIMA (Seasonal)", "Exponential Smoothing (ETS)"])

st.sidebar.header("3. Model Parameters")

# Conditional UI: Only show sliders relevant to the selected model
if model_type in ["ARIMA", "SARIMA (Seasonal)"]:
    st.sidebar.markdown("**Base ARIMA Parameters**")
    p = st.sidebar.slider("p (Lag/AutoRegressive)", 0, 10, 5)
    d = st.sidebar.slider("d (Differencing)", 0, 2, 1)
    q = st.sidebar.slider("q (Moving Average)", 0, 10, 0)

if model_type == "SARIMA (Seasonal)":
    st.sidebar.markdown("**Seasonal Parameters**")
    P = st.sidebar.slider("P (Seasonal AR)", 0, 3, 0)
    D = st.sidebar.slider("D (Seasonal Diff)", 0, 1, 0)
    Q = st.sidebar.slider("Q (Seasonal MA)", 0, 3, 0)
    s = st.sidebar.selectbox("Seasonality Cycle (s)", [5, 12, 21], format_func=lambda x: f"{x} (e.g., {'trading week' if x==5 else 'months' if x==12 else 'trading month'})")

if model_type == "Exponential Smoothing (ETS)":
    st.sidebar.markdown("**Holt-Winters Parameters**")
    trend = st.sidebar.selectbox("Trend Type", ["add", "mul", None], index=0)
    seasonal = st.sidebar.selectbox("Seasonal Type", ["add", "mul", None], index=0)
    seasonal_periods = st.sidebar.slider("Seasonal Periods (Days)", 2, 30, 5)

st.sidebar.header("4. Forecasting")
days_to_predict = st.sidebar.slider("Days to Forecast", 1, 90, 14)

# --- DATA FETCHING ---
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True)
    return data

if start_date >= end_date:
    st.error("Error: Start Date must be before End Date.")
else:
    data_load_state = st.text('Loading market data...')
    df = load_data(ticker, start_date, end_date)
    
    if df.empty:
        st.error(f"No data found for {ticker} in this date range. Check the ticker symbol.")
    else:
        data_load_state.text(f'Data for {ticker} loaded successfully!')

        fig_raw = go.Figure()
        fig_raw.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Historical Close"))
        fig_raw.layout.update(title_text=f'Historical Time Series Data for {ticker}', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig_raw, use_container_width=True)

        # --- PREDICTION MODELS ---
        st.subheader(f"Forecast Results: {model_type}")
        df_train = df['Close'].dropna().values
        
        with st.spinner(f"Fitting {model_type} model..."):
            try:
                # Dynamically fit the chosen model
                if model_type == "ARIMA":
                    model = ARIMA(df_train, order=(p, d, q))
                    fitted_model = model.fit()
                    forecast = fitted_model.forecast(steps=days_to_predict)
                    summary_text = fitted_model.summary().as_text()
                    
                elif model_type == "SARIMA (Seasonal)":
                    model = SARIMAX(df_train, order=(p, d, q), seasonal_order=(P, D, Q, s))
                    fitted_model = model.fit(disp=False)
                    forecast = fitted_model.forecast(steps=days_to_predict)
                    summary_text = fitted_model.summary().as_text()
                    
                elif model_type == "Exponential Smoothing (ETS)":
                    model = ExponentialSmoothing(df_train, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
                    fitted_model = model.fit()
                    forecast = fitted_model.forecast(days_to_predict)
                    summary_text = "Exponential Smoothing models do not generate standard statsmodels summaries. Check the chart for fit."

                # Generate future dates
                last_date = df['Date'].iloc[-1]
                future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
                
                # Plot
                fig_pred = go.Figure()
                context_df = df.tail(100)
                
                fig_pred.add_trace(go.Scatter(x=context_df['Date'], y=context_df['Close'], name="Recent Actual", line=dict(color='blue')))
                fig_pred.add_trace(go.Scatter(x=future_dates, y=forecast, name=f"{model_type} Forecast", line=dict(color='red', width=3, dash='dot')))
                
                fig_pred.layout.update(
                    title_text=f'{days_to_predict}-Day Forecast using {model_type}',
                    xaxis_title="Date",
                    yaxis_title="Price"
                )
                st.plotly_chart(fig_pred, use_container_width=True)
                
                with st.expander(f"View Model Summary"):
                    st.text(summary_text)

            except Exception as e:
                st.error("The model failed to converge. This happens when the chosen parameters mathematically clash with the data (e.g., multiplicative seasonality on data with zero-values, or over-differencing). Try adjusting the sliders.")
                st.exception(e)
