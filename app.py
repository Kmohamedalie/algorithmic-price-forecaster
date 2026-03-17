import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from datetime import date, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
import re
import warnings

# Ignore statistical convergence warnings in the UI
warnings.filterwarnings("ignore")

# --- UI SETUP ---
st.set_page_config(page_title="Ultimate Stock Predictor", layout="wide")
st.title("📈 The Ultimate Multi-Model Stock Predictor")
st.markdown("Compare Classic Statistical models against Modern Algorithmic approaches.")

# --- SIDEBAR (Global Settings) ---
st.sidebar.header("Global Data Configuration")

raw_ticker_input = st.sidebar.text_input("Stock Ticker (e.g., AAPL)", "AAPL").upper()
ticker = re.split(r'[,\s]+', raw_ticker_input)[0].strip()

start_date = st.sidebar.date_input("Start Date", date.today() - timedelta(days=365*2))
end_date = st.sidebar.date_input("End Date", date.today())

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
    st.sidebar.error("Error: Start Date must be before End Date.")
else:
    df = load_data(ticker, start_date, end_date)
    
    if df.empty:
        st.error(f"No data found for {ticker} in this date range. Check the ticker symbol.")
    else:
        # Plot raw data at the top so it's always visible
        fig_raw = go.Figure()
        fig_raw.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Historical Close"))
        fig_raw.layout.update(title_text=f'Historical Time Series Data for {ticker}', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig_raw, use_container_width=True)

        # Extract training data
        df_train_values = df['Close'].dropna().values

        # --- TABS SETUP ---
        tab1, tab2 = st.tabs(["📊 Classic Statistical Models", "🤖 Machine Learning & Prophet"])

        # ==========================================
        # TAB 1: STATISTICAL MODELS
        # ==========================================
        with tab1:
            st.subheader("Statistical Forecasting")
            model_type_stat = st.selectbox("Choose a Statistical Model", 
                                           ["ARIMA", "SARIMA (Seasonal)", "Exponential Smoothing (ETS)"])
            
            # Model Specific Parameters inside the tab
            st.markdown("##### Model Parameters")
            col1, col2, col3 = st.columns(3)
            
            if model_type_stat in ["ARIMA", "SARIMA (Seasonal)"]:
                with col1:
                    p = st.slider("p (AutoRegressive)", 0, 10, 5)
                with col2:
                    d = st.slider("d (Differencing)", 0, 2, 1)
                with col3:
                    q = st.slider("q (Moving Average)", 0, 10, 0)

            if model_type_stat == "SARIMA (Seasonal)":
                st.markdown("##### Seasonal Parameters")
                scol1, scol2, scol3, scol4 = st.columns(4)
                with scol1: P = st.slider("P (Seasonal AR)", 0, 3, 0)
                with scol2: D = st.slider("D (Seasonal Diff)", 0, 1, 0)
                with scol3: Q = st.slider("Q (Seasonal MA)", 0, 3, 0)
                with scol4: s = st.selectbox("Seasonality Cycle (s)", [5, 12, 21])

            if model_type_stat == "Exponential Smoothing (ETS)":
                ecol1, ecol2, ecol3 = st.columns(3)
                with ecol1: trend = st.selectbox("Trend Type", ["add", "mul", None], index=0)
                with ecol2: seasonal = st.selectbox("Seasonal Type", ["add", "mul", None], index=0)
                with ecol3: seasonal_periods = st.slider("Seasonal Periods", 2, 30, 5)

            with st.spinner(f"Fitting {model_type_stat} model..."):
                try:
                    if model_type_stat == "ARIMA":
                        model = ARIMA(df_train_values, order=(p, d, q))
                        fitted_model = model.fit()
                        forecast = fitted_model.forecast(steps=days_to_predict)
                        summary_text = fitted_model.summary().as_text()
                        
                    elif model_type_stat == "SARIMA (Seasonal)":
                        model = SARIMAX(df_train_values, order=(p, d, q), seasonal_order=(P, D, Q, s))
                        fitted_model = model.fit(disp=False)
                        forecast = fitted_model.forecast(steps=days_to_predict)
                        summary_text = fitted_model.summary().as_text()
                        
                    elif model_type_stat == "Exponential Smoothing (ETS)":
                        model = ExponentialSmoothing(df_train_values, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
                        fitted_model = model.fit()
                        forecast = fitted_model.forecast(days_to_predict)
                        summary_text = "Check the chart for fit."

                    last_date = df['Date'].iloc[-1]
                    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
                    
                    fig_stat = go.Figure()
                    context_df = df.tail(100)
                    fig_stat.add_trace(go.Scatter(x=context_df['Date'], y=context_df['Close'], name="Recent Actual", line=dict(color='blue')))
                    fig_stat.add_trace(go.Scatter(x=future_dates, y=forecast, name=f"{model_type_stat} Forecast", line=dict(color='red', width=3, dash='dot')))
                    fig_stat.layout.update(title_text=f'{days_to_predict}-Day Forecast ({model_type_stat})')
                    st.plotly_chart(fig_stat, use_container_width=True)
                    
                    with st.expander("View Model Summary"):
                        st.text(summary_text)

                except Exception as e:
                    st.error("Model failed to converge. Try adjusting the parameters.")

        # ==========================================
        # TAB 2: MACHINE LEARNING & PROPHET
        # ==========================================
        with tab2:
            st.subheader("Modern Algorithmic Forecasting")
            model_type_ml = st.selectbox("Choose Algorithmic Model", ["Facebook Prophet", "Machine Learning (Random Forest)"])
            
            with st.spinner(f"Running {model_type_ml}..."):
                if model_type_ml == "Facebook Prophet":
                    df_prophet = df[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
                    m = Prophet(daily_seasonality=True)
                    m.fit(df_prophet)
                    future = m.make_future_dataframe(periods=days_to_predict)
                    prophet_forecast = m.predict(future)
                    
                    fig_proph = go.Figure()
                    fig_proph.add_trace(go.Scatter(x=df_prophet['ds'].tail(100), y=df_prophet['y'].tail(100), name="Actual", line=dict(color='blue')))
                    
                    # Plot future only for clarity
                    future_only = prophet_forecast.tail(days_to_predict)
                    fig_proph.add_trace(go.Scatter(x=future_only['ds'], y=future_only['yhat'], name="Predicted", line=dict(color='orange')))
                    fig_proph.add_trace(go.Scatter(x=future_only['ds'], y=future_only['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
                    fig_proph.add_trace(go.Scatter(x=future_only['ds'], y=future_only['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', name='Confidence Interval', fillcolor='rgba(255, 165, 0, 0.2)'))
                    fig_proph.layout.update(title_text=f'Prophet Forecast for {ticker}')
                    st.plotly_chart(fig_proph, use_container_width=True)

                elif model_type_ml == "Machine Learning (Random Forest)":
                    df_rf = df[['Date', 'Close']].copy()
                    df_rf.dropna(subset=['Close'], inplace=True)
                    df_rf['Day'] = df_rf['Date'].dt.day
                    df_rf['Month'] = df_rf['Date'].dt.month
                    df_rf['Year'] = df_rf['Date'].dt.year
                    df_rf['DayOfWeek'] = df_rf['Date'].dt.dayofweek
                    
                    X = df_rf[['Day', 'Month', 'Year', 'DayOfWeek']]
                    y = df_rf['Close']
                    
                    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf_model.fit(X, y)
                    
                    last_date = df['Date'].iloc[-1]
                    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
                    future_df = pd.DataFrame({'Date': future_dates})
                    future_df['Day'] = future_df['Date'].dt.day
                    future_df['Month'] = future_df['Date'].dt.month
                    future_df['Year'] = future_df['Date'].dt.year
                    future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek
                    
                    rf_forecast = rf_model.predict(future_df[['Day', 'Month', 'Year', 'DayOfWeek']])
                    
                    fig_rf = go.Figure()
                    context_df = df.tail(100)
                    fig_rf.add_trace(go.Scatter(x=context_df['Date'], y=context_df['Close'], name="Recent Actual", line=dict(color='blue')))
                    fig_rf.add_trace(go.Scatter(x=future_dates, y=rf_forecast, name="Predicted (Random Forest)", line=dict(color='green', dash='dot')))
                    fig_rf.layout.update(title_text=f'Random Forest Forecast for {ticker}')
                    st.plotly_chart(fig_rf, use_container_width=True)
                    
                    st.warning("Note: This simple ML model uses date-derived features. It does not use autoregression (lagging past prices).")
