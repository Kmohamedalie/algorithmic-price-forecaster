import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from datetime import date, timedelta
from statsmodels.tsa.arima.model import ARIMA
import re
import warnings

# Ignore statistical convergence warnings in the UI
warnings.filterwarnings("ignore")

# --- UI SETUP ---
st.set_page_config(page_title="ARIMA Stock Predictor", layout="wide")
st.title("📈 Statistical Stock Forecasting (ARIMA)")
st.markdown("Use the Autoregressive Integrated Moving Average (ARIMA) model to forecast short-term stock trends.")

# --- SIDEBAR ---
st.sidebar.header("1. Data Configuration")

# Ticker Input
raw_ticker_input = st.sidebar.text_input("Stock Ticker (e.g., AAPL)", "AAPL").upper()
ticker = re.split(r'[,\s]+', raw_ticker_input)[0].strip()

# Date Range Picker
start_date = st.sidebar.date_input("Start Date", date.today() - timedelta(days=365*2)) # Default: 2 years ago
end_date = st.sidebar.date_input("End Date", date.today())

st.sidebar.header("2. ARIMA Parameters (p, d, q)")
st.sidebar.markdown("Tune these to improve model fit.")
p = st.sidebar.slider("p (Lag/AutoRegressive)", 0, 10, 5)
d = st.sidebar.slider("d (Differencing)", 0, 2, 1)
q = st.sidebar.slider("q (Moving Average)", 0, 10, 0)

st.sidebar.header("3. Forecasting")
days_to_predict = st.sidebar.slider("Days to Forecast", 1, 90, 14)

# --- DATA FETCHING ---
@st.cache_data
def load_data(ticker, start, end):
    # Fetch data based on user-selected dates
    data = yf.download(ticker, start=start, end=end)
    
    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    data.reset_index(inplace=True)
    return data

# Verify dates make sense
if start_date >= end_date:
    st.error("Error: Start Date must be before End Date.")
else:
    data_load_state = st.text('Loading market data...')
    df = load_data(ticker, start_date, end_date)
    
    if df.empty:
        st.error(f"No data found for {ticker} in this date range. Check the ticker symbol.")
    else:
        data_load_state.text(f'Data for {ticker} loaded successfully!')

        # --- PLOTTING RAW DATA ---
        fig_raw = go.Figure()
        fig_raw.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Historical Close"))
        fig_raw.layout.update(title_text=f'Historical Time Series Data for {ticker}', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig_raw, use_container_width=True)

        # --- ARIMA PREDICTION MODEL ---
        st.subheader("Forecast Results")
        
        # Extract the closing prices and drop any missing data
        df_train = df['Close'].dropna().values
        
        with st.spinner("Fitting ARIMA model... This may take a moment depending on the parameters."):
            try:
                # Fit the model using user-selected p, d, q
                model = ARIMA(df_train, order=(p, d, q))
                fitted_model = model.fit()
                
                # Predict future values
                forecast = fitted_model.forecast(steps=days_to_predict)
                
                # Generate future dates for the x-axis
                last_date = df['Date'].iloc[-1]
                future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
                
                # --- PLOTTING FORECAST ---
                fig_pred = go.Figure()
                # Plot the last 100 days of actual data for visual context instead of the whole history
                context_df = df.tail(100)
                
                fig_pred.add_trace(go.Scatter(x=context_df['Date'], y=context_df['Close'], name="Recent Actual", line=dict(color='blue')))
                fig_pred.add_trace(go.Scatter(x=future_dates, y=forecast, name="ARIMA Forecast", line=dict(color='red', width=3, dash='dot')))
                
                fig_pred.layout.update(
                    title_text=f'{days_to_predict}-Day Forecast using ARIMA({p},{d},{q})',
                    xaxis_title="Date",
                    yaxis_title="Price"
                )
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Show model summary statistics
                with st.expander("View ARIMA Model Summary (Advanced)"):
                    st.text(fitted_model.summary().as_text())

            except Exception as e:
                st.error(f"The ARIMA model failed to converge with parameters ({p},{d},{q}). This is common in statistical modeling when the parameters don't fit the data shape. Try adjusting the p, d, or q sliders, or selecting a different timeframe.")
                st.exception(e)
