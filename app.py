import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import date, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import re
import warnings
from fpdf import FPDF
import tempfile

# Ignore statistical convergence warnings in the UI
warnings.filterwarnings("ignore")

# --- UI SETUP ---
st.set_page_config(page_title="Ultimate Market Predictor", layout="wide")
st.title("📈 The Ultimate Multi-Model Market Predictor")
st.markdown("Forecast Stocks, Currencies, Crypto, and Commodities using Stats and Machine Learning.")

# Initialize Session State for the Macro model so data survives button clicks
if 'macro_results' not in st.session_state:
    st.session_state.macro_results = None

# --- SIDEBAR (Global Settings) ---
st.sidebar.header("Global Data Configuration")

raw_ticker_input = st.sidebar.text_input("Target Ticker (e.g., AAPL, EURUSD=X, BTC-USD)", "AAPL").upper()
ticker = re.split(r'[,\s]+', raw_ticker_input)[0].strip()

start_date = st.sidebar.date_input("Start Date", date.today() - timedelta(days=365*3))
end_date = st.sidebar.date_input("End Date", date.today())

days_to_predict = st.sidebar.slider("Days to Forecast", 1, 90, 14)

# When global settings change, clear the old saved results so we don't download outdated data
def clear_state():
    st.session_state.macro_results = None

st.sidebar.button("Reset Dashboard", on_click=clear_state)

# --- DATA FETCHING ---
@st.cache_data(ttl=3600)
def load_data(ticker_symbol, start, end):
    start_str = start.strftime('%Y-%m-%d')
    end_str = end.strftime('%Y-%m-%d')
    data = yf.download(ticker_symbol, start=start_str, end=end_str)
    
    if not data.empty and isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    data.reset_index(inplace=True)
    return data

if start_date >= end_date:
    st.sidebar.error("Error: Start Date must be before End Date.")
else:
    with st.spinner('Fetching market data...'):
        df = load_data(ticker, start_date, end_date)
    
    if df.empty:
        st.error(f"No data found for {ticker} in this date range.")
    else:
        fig_raw = go.Figure()
        fig_raw.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Historical Close"))
        fig_raw.layout.update(title_text=f'Historical Price Data for {ticker}', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig_raw, use_container_width=True)

        df_train_values = df['Close'].dropna().values

        # --- TABS SETUP ---
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Statistical", 
            "🤖 ML & Prophet", 
            "🌍 Macro SARIMAX", 
            "📖 Asset Class Guide"
        ])

        # ==========================================
        # TAB 1: STATISTICAL MODELS
        # ==========================================
        with tab1:
            st.subheader("Classic Statistical Forecasting")
            model_type_stat = st.selectbox("Choose a Statistical Model", 
                                           ["ARIMA", "SARIMA (Seasonal)", "Exponential Smoothing (ETS)"])
            
            st.markdown("##### Model Parameters")
            col1, col2, col3 = st.columns(3)
            if model_type_stat in ["ARIMA", "SARIMA (Seasonal)"]:
                with col1: p = st.slider("p (AutoRegressive)", 0, 10, 5, key="t1_p")
                with col2: d = st.slider("d (Differencing)", 0, 2, 1, key="t1_d")
                with col3: q = st.slider("q (Moving Average)", 0, 10, 0, key="t1_q")

            if model_type_stat == "SARIMA (Seasonal)":
                st.markdown("##### Seasonal Parameters")
                scol1, scol2, scol3, scol4 = st.columns(4)
                with scol1: P = st.slider("P (Seasonal AR)", 0, 3, 0, key="t1_P")
                with scol2: D = st.slider("D (Seasonal Diff)", 0, 1, 0, key="t1_D")
                with scol3: Q = st.slider("Q (Seasonal MA)", 0, 3, 0, key="t1_Q")
                with scol4: s = st.selectbox("Seasonality Cycle (s)", [5, 12, 21], key="t1_s")

            if model_type_stat == "Exponential Smoothing (ETS)":
                ecol1, ecol2, ecol3 = st.columns(3)
                with ecol1: trend = st.selectbox("Trend Type", ["add", "mul", None], index=0)
                with ecol2: seasonal = st.selectbox("Seasonal Type", ["add", "mul", None], index=0)
                with ecol3: seasonal_periods = st.slider("Seasonal Periods", 2, 30, 5)

            if st.button(f"Run {model_type_stat} Model"):
                with st.spinner(f"Fitting {model_type_stat} model..."):
                    try:
                        if model_type_stat == "ARIMA":
                            model = ARIMA(df_train_values, order=(p, d, q))
                            fitted_model = model.fit()
                            forecast = fitted_model.forecast(steps=days_to_predict)
                            summary_text = fitted_model.summary().as_text()
                            # Calculate RMSE (Skipping first 30 days to avoid startup noise)
                            rmse = np.sqrt(mean_squared_error(df_train_values[30:], fitted_model.fittedvalues[30:]))

                        elif model_type_stat == "SARIMA (Seasonal)":
                            model = SARIMAX(df_train_values, order=(p, d, q), seasonal_order=(P, D, Q, s))
                            fitted_model = model.fit(disp=False)
                            forecast = fitted_model.forecast(steps=days_to_predict)
                            summary_text = fitted_model.summary().as_text()
                            rmse = np.sqrt(mean_squared_error(df_train_values[30:], fitted_model.fittedvalues[30:]))

                        elif model_type_stat == "Exponential Smoothing (ETS)":
                            model = ExponentialSmoothing(df_train_values, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
                            fitted_model = model.fit()
                            forecast = fitted_model.forecast(days_to_predict)
                            summary_text = "Note: Exponential smoothing calculates a visual fit and does not generate standard P-value summary tables."
                            rmse = np.sqrt(mean_squared_error(df_train_values[30:], fitted_model.fittedvalues[30:]))

                        # Display RMSE Metric
                        st.metric(label=f"Historical Accuracy Grade (RMSE)", value=round(rmse, 4), help="Root Mean Square Error. Lower is better! This shows how far off the model was on average during the historical training data.")

                        last_date = df['Date'].iloc[-1]
                        future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
                        
                        fig_stat = go.Figure()
                        fig_stat.add_trace(go.Scatter(x=df['Date'].tail(150), y=df['Close'].tail(150), name="Recent Actual", line=dict(color='blue')))
                        fig_stat.add_trace(go.Scatter(x=future_dates, y=forecast, name=f"{model_type_stat} Forecast", line=dict(color='red', width=3, dash='dot')))
                        fig_stat.layout.update(title_text=f'{days_to_predict}-Day Forecast ({model_type_stat})')
                        st.plotly_chart(fig_stat, use_container_width=True)
                        
                        with st.expander("View Statistical Model Summary"):
                            st.text(summary_text)
                    except Exception as e:
                        st.error("Model failed to converge. Try adjusting parameters.")

        # ==========================================
        # TAB 2: MACHINE LEARNING & PROPHET
        # ==========================================
        with tab2:
            st.subheader("Modern Algorithmic Forecasting")
            model_type_ml = st.selectbox("Choose Algorithmic Model", ["Facebook Prophet", "Machine Learning (Random Forest)"])
            
            if st.button(f"Run {model_type_ml} Model"):
                with st.spinner(f"Running {model_type_ml}..."):
                    if model_type_ml == "Facebook Prophet":
                        df_prophet = df[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
                        m = Prophet(daily_seasonality=True)
                        m.fit(df_prophet)
                        future = m.make_future_dataframe(periods=days_to_predict)
                        prophet_forecast = m.predict(future)
                        
                        # Calculate RMSE
                        historical_predictions = prophet_forecast['yhat'][:len(df_prophet)]
                        rmse = np.sqrt(mean_squared_error(df_prophet['y'], historical_predictions))
                        st.metric(label=f"Historical Accuracy Grade (RMSE)", value=round(rmse, 4))
                        
                        fig_proph = go.Figure()
                        fig_proph.add_trace(go.Scatter(x=df_prophet['ds'].tail(150), y=df_prophet['y'].tail(150), name="Actual", line=dict(color='blue')))
                        future_only = prophet_forecast.tail(days_to_predict)
                        fig_proph.add_trace(go.Scatter(x=future_only['ds'], y=future_only['yhat'], name="Predicted", line=dict(color='orange', width=3, dash='dot')))
                        fig_proph.add_trace(go.Scatter(x=future_only['ds'], y=future_only['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
                        fig_proph.add_trace(go.Scatter(x=future_only['ds'], y=future_only['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', fillcolor='rgba(255, 165, 0, 0.2)'))
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
                        
                        # Calculate RMSE
                        historical_predictions = rf_model.predict(X)
                        rmse = np.sqrt(mean_squared_error(y, historical_predictions))
                        st.metric(label=f"Historical Accuracy Grade (RMSE)", value=round(rmse, 4))
                        
                        last_date = df['Date'].iloc[-1]
                        future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
                        future_df = pd.DataFrame({'Date': future_dates})
                        future_df['Day'] = future_df['Date'].dt.day
                        future_df['Month'] = future_df['Date'].dt.month
                        future_df['Year'] = future_df['Date'].dt.year
                        future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek
                        
                        rf_forecast = rf_model.predict(future_df[['Day', 'Month', 'Year', 'DayOfWeek']])
                        
                        fig_rf = go.Figure()
                        fig_rf.add_trace(go.Scatter(x=df['Date'].tail(150), y=df['Close'].tail(150), name="Recent Actual", line=dict(color='blue')))
                        fig_rf.add_trace(go.Scatter(x=future_dates, y=rf_forecast, name="Predicted", line=dict(color='green', width=3, dash='dot')))
                        fig_rf.layout.update(title_text=f'Random Forest Forecast for {ticker}')
                        st.plotly_chart(fig_rf, use_container_width=True)
                        
                with st.expander("Why is there no Model Summary here?"):
                    st.info("Unlike ARIMA or SARIMA, Machine Learning algorithms (like Random Forest) and Prophet are not traditional statistical equations. They do not calculate p-values, z-scores, or standard errors for individual variables. Therefore, there is no statistical summary table to display.")

        # ==========================================
        # TAB 3: MACROECONOMIC SARIMAX
        # ==========================================
        with tab3:
            st.subheader("Multivariate SARIMAX (with Exogenous Variables)")
            st.markdown("Predict your target asset by mathematically factoring in the live movements of a macroeconomic indicator.")
            
            macro_dict = {
                "10-Year US Treasury Yield (^TNX)": "^TNX",
                "US Dollar Index (DX-Y.NYB)": "DX-Y.NYB",
                "Crude Oil (CL=F)": "CL=F"
            }
            macro_choice = st.selectbox("Select Macroeconomic Indicator (Exogenous Variable)", list(macro_dict.keys()), on_change=clear_state)
            macro_ticker = macro_dict[macro_choice]

            st.markdown("##### Target Asset SARIMAX Parameters")
            mcol1, mcol2, mcol3 = st.columns(3)
            with mcol1: mp = st.slider("p (AutoRegressive)", 0, 10, 5, key="t3_p", on_change=clear_state)
            with mcol2: md = st.slider("d (Differencing)", 0, 2, 1, key="t3_d", on_change=clear_state)
            with mcol3: mq = st.slider("q (Moving Average)", 0, 10, 0, key="t3_q", on_change=clear_state)

            if st.button("Run Macro SARIMAX Model"):
                with st.spinner(f"Fetching data and aligning dates..."):
                    macro_df = load_data(macro_ticker, start_date, end_date)
                    
                    if macro_df.empty:
                        st.error(f"Could not fetch data for {macro_ticker}.")
                    else:
                        merged_df = pd.merge(df[['Date', 'Close']], macro_df[['Date', 'Close']], on='Date', suffixes=('_Target', '_Macro'))
                        merged_df.dropna(inplace=True)

                        if merged_df.empty:
                            st.error("Data alignment failed. The target and macro indicator do not share enough trading days.")
                        else:
                            endog = merged_df['Close_Target'].values
                            exog = merged_df['Close_Macro'].values

                            try:
                                macro_model = SARIMAX(endog, exog=exog, order=(mp, md, mq))
                                macro_fitted = macro_model.fit(disp=False)
                                
                                future_exog = np.repeat(exog[-1], days_to_predict)
                                macro_forecast = macro_fitted.forecast(steps=days_to_predict, exog=future_exog)
                                
                                # Calculate RMSE (Skipping first 30 days)
                                rmse = np.sqrt(mean_squared_error(endog[30:], macro_fitted.fittedvalues[30:]))

                                last_date = merged_df['Date'].iloc[-1]
                                future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]

                                # 1. Create Chart
                                fig_macro = go.Figure()
                                fig_macro.add_trace(go.Scatter(x=merged_df['Date'].tail(150), y=merged_df['Close_Target'].tail(150), name="Target Actual", line=dict(color='blue')))
                                fig_macro.add_trace(go.Scatter(x=future_dates, y=macro_forecast, name="Macro SARIMAX Forecast", line=dict(color='purple', width=3, dash='dot')))
                                fig_macro.layout.update(title_text=f'{days_to_predict}-Day Target Forecast (Driven by {macro_ticker})')

                                # 2. Generate CSV Bytes
                                forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_Target_Price': macro_forecast, 'Exogenous_Variable_Assumption': future_exog})
                                csv_data = forecast_df.to_csv(index=False).encode('utf-8')
                                
                                # 3. Generate PDF Bytes
                                summary_text = macro_fitted.summary().as_text()
                                pdf = FPDF()
                                pdf.add_page()
                                pdf.set_font("Courier", size=8)
                                for line in summary_text.split('\n'):
                                    clean_line = line.encode('latin-1', 'replace').decode('latin-1')
                                    pdf.cell(0, 4, txt=clean_line, ln=1)
                                    
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                    pdf.output(tmp.name)
                                    with open(tmp.name, "rb") as f:
                                        pdf_bytes = f.read()

                                # SAVE TO SESSION STATE
                                st.session_state.macro_results = {
                                    'fig': fig_macro,
                                    'csv': csv_data,
                                    'pdf': pdf_bytes,
                                    'summary': summary_text,
                                    'rmse': rmse
                                }
                                    
                            except Exception as e:
                                st.error("SARIMAX failed to converge. Try adjusting the sliders.")

            # Display the UI using data from session_state
            if st.session_state.macro_results is not None:
                res = st.session_state.macro_results
                
                st.metric(label=f"Historical Accuracy Grade (RMSE)", value=round(res['rmse'], 4), help="Root Mean Square Error. Lower is better!")
                st.plotly_chart(res['fig'], use_container_width=True)

                st.markdown("### 📥 Export Your Results")
                col_csv, col_pdf = st.columns(2)
                
                with col_csv:
                    st.download_button(
                        label="Download Forecast Data (.csv)",
                        data=res['csv'],
                        file_name=f"{ticker}_SARIMAX_Forecast.csv",
                        mime="text/csv",
                        help="Download the predicted dates and prices to open in Excel."
                    )
                with col_pdf:
                    st.download_button(
                        label="Download Model Summary (.pdf)",
                        data=res['pdf'],
                        file_name=f"{ticker}_SARIMAX_Summary.pdf",
                        mime="application/pdf",
                        help="Download the dense statistical report containing your AIC scores and P-Values."
                    )
                
                with st.expander("View Macro SARIMAX Summary"):
                    st.text(res['summary'])

        # ==========================================
        # TAB 4: MARKET GUIDE
        # ==========================================
        with tab4:
            st.subheader("📖 Guide to Asset Classes & Tickers")
            st.markdown("Different assets behave completely differently. Understanding *what* drives an asset helps you choose the right forecasting model.")
            
            st.markdown("### 🏢 Stocks & Equities")
            st.markdown("""
            * **What drives them:** Company earnings reports, CEO changes, product launches, and general stock market sentiment.
            * **Best Models:** ARIMA for short-term momentum; Random Forest and Prophet for baseline trend identification. 
            * **Example Tickers:** `AAPL` (Apple), `TSLA` (Tesla), `SPY` (S&P 500 Index)
            """)

            st.markdown("### 💱 Currencies (Forex)")
            st.markdown("""
            * **What drives them:** Global macroeconomics. Interest rate hikes by central banks (like the US Fed), inflation rates, and international trade balances.
            * **Best Models:** **Macro SARIMAX** (using interest rates as exogenous variables).
            * **Example Tickers (Must end in =X):** `EURUSD=X` (Euro/US Dollar), `JPY=X` (Yen/US Dollar)
            """)

            st.markdown("### 🪙 Cryptocurrencies")
            st.markdown("""
            * **What drives them:** Network adoption, retail sentiment, regulatory news, and global liquidity (interest rates). They trade 24/7, meaning there are no weekend data gaps.
            * **Best Models:** **Facebook Prophet** is excellent at catching the 24/7 weekly seasonality of retail crypto traders. **Macro SARIMAX** is also powerful if you use the US Dollar Index or the NASDAQ as an outside variable, as Bitcoin often reacts to macro liquidity.
            * **Example Tickers (Must end in -USD):** `BTC-USD` (Bitcoin), `ETH-USD` (Ethereum), `SOL-USD` (Solana)
            """)

            st.markdown("### 📊 Market Indices & Real Estate (REITs)")
            st.markdown("""
            * **What drives them:** Broad macroeconomic health, collective corporate earnings, and interest rates (especially for Real Estate). 
            * **Best Models:** **Prophet** for long-term baseline trends of Indices. **Macro SARIMAX** for REITs, using the 10-Year Treasury Yield (`^TNX`) as an outside variable since real estate is highly sensitive to borrowing costs.
            * **Example Tickers:** `^GSPC` (S&P 500 Index), `^IXIC` (NASDAQ), `VNQ` (Vanguard Real Estate ETF)
            """)

            st.markdown("### 🛢️ Commodities & Precious Metals")
            st.markdown("""
            * **What drives them:** Physical supply/demand, geopolitical crises, and inverse correlations to the US Dollar.
            * **Best Models:** **Macro SARIMAX** is exceptional here. Try predicting Gold while feeding the model the US Treasury Yield as an outside factor.
            * **Example Tickers:** `GC=F` (Gold Futures), `SI=F` (Silver Futures), `CL=F` (Crude Oil)
            """)
