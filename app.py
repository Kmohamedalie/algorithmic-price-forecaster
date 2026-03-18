import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import date, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import scipy.optimize as sco
import re
import warnings
from fpdf import FPDF
import tempfile

warnings.filterwarnings("ignore")

# --- UI SETUP ---
st.set_page_config(page_title="Multi-Asset Quant Terminal", layout="wide")
st.title("📈 Multi-Asset Quantitative Strategy Terminal")

# Initialize Session States (Kept strictly false for auto-clearing)
if 'macro_results' not in st.session_state: st.session_state.macro_results = None
if 'corr_data' not in st.session_state: st.session_state.corr_data = None
if 'port_results' not in st.session_state: st.session_state.port_results = None

# --- SIDEBAR ---
st.sidebar.header("Global Configuration")
raw_ticker_input = st.sidebar.text_input("Target Ticker (For Tabs 1-4)", "AAPL").upper()
ticker = re.split(r'[,\s]+', raw_ticker_input)[0].strip()

start_date = st.sidebar.date_input("Start Date", date.today() - timedelta(days=365*3))
end_date = st.sidebar.date_input("End Date", date.today())
days_to_predict = st.sidebar.slider("Forecast Horizon", 1, 90, 14)

st.sidebar.markdown("---")
st.sidebar.subheader("Technical Overlays")
show_sma = st.sidebar.checkbox("Show SMA (50 & 200)", value=True)
show_rsi = st.sidebar.checkbox("Show RSI (14)", value=True)

# A manual reset button
if st.sidebar.button("🧹 Clear Saved Data"):
    st.session_state.macro_results = None
    st.session_state.corr_data = None
    st.session_state.port_results = None

# --- DATA ENGINE ---
@st.cache_data(ttl=3600)
def load_data(ticker_symbol, start, end):
    data = yf.download(ticker_symbol, start=start, end=end, progress=False)
    if not data.empty and isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True)
    return data

df = load_data(ticker, start_date, end_date)

if not df.empty:
    df_train_values = df['Close'].dropna().values

    # Calculate TA Indicators
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MAIN CHART
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3]) if show_rsi else go.Figure()
    
    main_trace = go.Scatter(x=df['Date'], y=df['Close'], name="Close Price", line=dict(color='#1f77b4'))
    if show_rsi:
        fig.add_trace(main_trace, row=1, col=1)
        if show_sma:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA50'], name="SMA 50", line=dict(width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA200'], name="SMA 200", line=dict(width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name="RSI", line=dict(color='orange')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", row=2, col=1, line_color="red")
        fig.add_hline(y=30, line_dash="dot", row=2, col=1, line_color="green")
    else:
        fig.add_trace(main_trace)
        if show_sma:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA50'], name="SMA 50"))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA200'], name="SMA 200"))
    
    st.plotly_chart(fig, use_container_width=True)

    # --- TABS ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Statistical", 
        "🤖 ML & Prophet", 
        "🌍 Macro SARIMAX", 
        "🔍 Macro Scanner", 
        "⚖️ Portfolio Optimizer", 
        "📖 Guide"
    ])

    # ==========================================
    # TAB 1: STATISTICAL MODELS (Restored Tuning)
    # ==========================================
    with tab1:
        st.subheader("Classic Statistical Forecasting")
        model_type_stat = st.selectbox("Choose a Statistical Model", ["ARIMA", "SARIMA (Seasonal)", "Exponential Smoothing (ETS)"])
        
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
            with st.spinner(f"Fitting {model_type_stat}..."):
                try:
                    skip_days = 30 if len(df_train_values) > 60 else 0 
                    if model_type_stat == "ARIMA":
                        model = ARIMA(df_train_values, order=(p, d, q))
                        fitted_model = model.fit()
                        forecast = fitted_model.forecast(steps=days_to_predict)
                        summary_text = fitted_model.summary().as_text()
                        rmse = np.sqrt(mean_squared_error(df_train_values[skip_days:], fitted_model.fittedvalues[skip_days:]))
                    elif model_type_stat == "SARIMA (Seasonal)":
                        model = SARIMAX(df_train_values, order=(p, d, q), seasonal_order=(P, D, Q, s))
                        fitted_model = model.fit(disp=False)
                        forecast = fitted_model.forecast(steps=days_to_predict)
                        summary_text = fitted_model.summary().as_text()
                        rmse = np.sqrt(mean_squared_error(df_train_values[skip_days:], fitted_model.fittedvalues[skip_days:]))
                    elif model_type_stat == "Exponential Smoothing (ETS)":
                        model = ExponentialSmoothing(df_train_values, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
                        fitted_model = model.fit()
                        forecast = fitted_model.forecast(days_to_predict)
                        summary_text = "Note: Exponential smoothing calculates a visual fit and does not generate standard P-value summary tables."
                        rmse = np.sqrt(mean_squared_error(df_train_values[skip_days:], fitted_model.fittedvalues[skip_days:]))

                    st.metric(label=f"Historical Accuracy Grade (RMSE)", value=round(rmse, 4))
                    
                    last_date = df['Date'].iloc[-1]
                    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
                    
                    fig_stat = go.Figure()
                    fig_stat.add_trace(go.Scatter(x=df['Date'].tail(150), y=df['Close'].tail(150), name="Actual"))
                    fig_stat.add_trace(go.Scatter(x=future_dates, y=forecast, name="Forecast", line=dict(color='red', width=3, dash='dot')))
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
                    
                    rmse = np.sqrt(mean_squared_error(df_prophet['y'], prophet_forecast['yhat'][:len(df_prophet)]))
                    st.metric(label=f"Historical Accuracy Grade (RMSE)", value=round(rmse, 4))
                    
                    fig_proph = go.Figure()
                    fig_proph.add_trace(go.Scatter(x=df_prophet['ds'].tail(150), y=df_prophet['y'].tail(150), name="Actual", line=dict(color='blue')))
                    future_only = prophet_forecast.tail(days_to_predict)
                    fig_proph.add_trace(go.Scatter(x=future_only['ds'], y=future_only['yhat'], name="Predicted", line=dict(color='orange', width=3, dash='dot')))
                    st.plotly_chart(fig_proph, use_container_width=True)

                elif model_type_ml == "Machine Learning (Random Forest)":
                    df_rf = df[['Date', 'Close']].copy().dropna()
                    df_rf['Day'] = df_rf['Date'].dt.day
                    df_rf['Month'] = df_rf['Date'].dt.month
                    df_rf['Year'] = df_rf['Date'].dt.year
                    
                    X = df_rf[['Day', 'Month', 'Year']]
                    y = df_rf['Close']
                    rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
                    
                    rmse = np.sqrt(mean_squared_error(y, rf_model.predict(X)))
                    st.metric(label=f"Historical Accuracy Grade (RMSE)", value=round(rmse, 4))
                    
                    last_date = df['Date'].iloc[-1]
                    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
                    future_df = pd.DataFrame({'Date': future_dates, 'Day': [d.day for d in future_dates], 'Month': [d.month for d in future_dates], 'Year': [d.year for d in future_dates]})
                    
                    rf_forecast = rf_model.predict(future_df[['Day', 'Month', 'Year']])
                    
                    fig_rf = go.Figure()
                    fig_rf.add_trace(go.Scatter(x=df['Date'].tail(150), y=df['Close'].tail(150), name="Actual"))
                    fig_rf.add_trace(go.Scatter(x=future_dates, y=rf_forecast, name="Predicted", line=dict(color='green', width=3, dash='dot')))
                    st.plotly_chart(fig_rf, use_container_width=True)

    # ==========================================
    # TAB 3: MACROECONOMIC SARIMAX (Restored PDF/CSV/Sliders)
    # ==========================================
    with tab3:
        st.subheader("Multivariate SARIMAX (with Exogenous Variables)")
        
        macro_dict = {"10-Year US Treasury Yield (^TNX)": "^TNX", "US Dollar Index (DX-Y.NYB)": "DX-Y.NYB", "Crude Oil (CL=F)": "CL=F"}
        macro_choice = st.selectbox("Select Exogenous Variable", list(macro_dict.keys()))
        macro_ticker = macro_dict[macro_choice]

        st.markdown("##### Target Asset SARIMAX Parameters")
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1: mp = st.slider("p (AutoRegressive)", 0, 10, 5, key="t3_p")
        with mcol2: md = st.slider("d (Differencing)", 0, 2, 1, key="t3_d")
        with mcol3: mq = st.slider("q (Moving Average)", 0, 10, 0, key="t3_q")

        if st.button("Run Macro SARIMAX Model"):
            with st.spinner("Fetching data and running SARIMAX..."):
                macro_df = load_data(macro_ticker, start_date, end_date)
                if macro_df.empty:
                    st.error(f"Could not fetch data for {macro_ticker}.")
                else:
                    merged_df = pd.merge(df[['Date', 'Close']], macro_df[['Date', 'Close']], on='Date', suffixes=('_Target', '_Macro')).dropna()
                    endog = merged_df['Close_Target'].values
                    exog = merged_df['Close_Macro'].values

                    try:
                        macro_model = SARIMAX(endog, exog=exog, order=(mp, md, mq))
                        macro_fitted = macro_model.fit(disp=False)
                        
                        future_exog = np.repeat(exog[-1], days_to_predict)
                        macro_forecast = macro_fitted.forecast(steps=days_to_predict, exog=future_exog)
                        
                        skip_days = 30 if len(endog) > 60 else 0
                        rmse = np.sqrt(mean_squared_error(endog[skip_days:], macro_fitted.fittedvalues[skip_days:]))

                        last_date = merged_df['Date'].iloc[-1]
                        future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]

                        # Chart
                        fig_macro = go.Figure()
                        fig_macro.add_trace(go.Scatter(x=merged_df['Date'].tail(150), y=merged_df['Close_Target'].tail(150), name="Target Actual", line=dict(color='blue')))
                        fig_macro.add_trace(go.Scatter(x=future_dates, y=macro_forecast, name="Macro Forecast", line=dict(color='purple', width=3, dash='dot')))
                        
                        # CSV Data
                        forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_Target': macro_forecast, 'Exogenous_Assumption': future_exog})
                        csv_data = forecast_df.to_csv(index=False).encode('utf-8')
                        
                        # PDF Data
                        summary_text = macro_fitted.summary().as_text()
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Courier", size=8)
                        for line in summary_text.split('\n'):
                            clean_line = line.encode('latin-1', 'replace').decode('latin-1')
                            pdf.cell(0, 4, txt=clean_line, ln=1)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            pdf.output(tmp.name)
                            with open(tmp.name, "rb") as f: pdf_bytes = f.read()

                        # Save to Session State
                        st.session_state.macro_results = {
                            'fig': fig_macro, 'csv': csv_data, 'pdf': pdf_bytes, 'summary': summary_text, 'rmse': rmse, 'ticker': ticker
                        }
                    except Exception as e:
                        st.error("SARIMAX failed to converge.")

        if st.session_state.macro_results is not None:
            res = st.session_state.macro_results
            if res.get('ticker') == ticker:
                st.metric(label="Historical Accuracy Grade (RMSE)", value=round(res['rmse'], 4))
                st.plotly_chart(res['fig'], use_container_width=True)

                st.markdown("### 📥 Export Your Results")
                col_csv, col_pdf = st.columns(2)
                with col_csv:
                    st.download_button("Download Forecast Data (.csv)", data=res['csv'], file_name=f"{ticker}_SARIMAX.csv", mime="text/csv")
                with col_pdf:
                    st.download_button("Download Model Summary (.pdf)", data=res['pdf'], file_name=f"{ticker}_SARIMAX.pdf", mime="application/pdf")
                with st.expander("View Macro SARIMAX Summary"):
                    st.text(res['summary'])
            else:
                st.info("Ticker changed. Click 'Run Macro SARIMAX Model' to generate new data.")

    # ==========================================
    # TAB 4: MACRO SCANNER
    # ==========================================
    with tab4:
        st.subheader("Global Macroeconomic Correlation")
        
        if st.button("Generate Correlation Heatmap"):
            with st.spinner("Scanning Global Markets..."):
                assets = {"Target Asset": ticker, "US Dollar": "DX-Y.NYB", "Gold": "GC=F", "Crude Oil": "CL=F", "10Y Yield": "^TNX", "S&P 500": "^GSPC"}
                corr_df = pd.DataFrame()
                for name, symbol in assets.items():
                    tmp_df = yf.download(symbol, start=start_date, end=end_date, progress=False)
                    if not tmp_df.empty and 'Close' in tmp_df:
                        close_col = tmp_df['Close'].iloc[:, 0] if isinstance(tmp_df['Close'], pd.DataFrame) else tmp_df['Close']
                        corr_df[name] = close_col
                
                corrs = corr_df.corr()
                fig_corr = go.Figure(data=go.Heatmap(z=corrs.values, x=corrs.columns, y=corrs.columns, colorscale='RdBu', zmin=-1, zmax=1))
                st.session_state.corr_data = fig_corr
                
        if st.session_state.corr_data:
            st.plotly_chart(st.session_state.corr_data, use_container_width=True)
            st.info("💡 **How to read this:** Use the darkest colored asset (closest to +1 or -1) as your exogenous variable in Tab 3!")

    # ==========================================
    # TAB 5: PORTFOLIO OPTIMIZER
    # ==========================================
    with tab5:
        st.subheader("⚖️ Modern Portfolio Theory (MPT)")
        multi_tickers = st.text_input("Portfolio Tickers (comma separated)", "AAPL, MSFT, GLD, BTC-USD").upper()
        
        if st.button("Optimize Weights"):
            with st.spinner("Calculating the Efficient Frontier..."):
                t_list = [x.strip() for x in multi_tickers.split(",")]
                if len(t_list) < 2:
                    st.error("You need at least 2 assets to build a portfolio!")
                else:
                    try:
                        port_data = yf.download(t_list, start=start_date, end=end_date)['Close'].dropna()
                        returns = port_data.pct_change().dropna()
                        mean_returns = returns.mean() * 252
                        cov_matrix = returns.cov() * 252
                        risk_free_rate = 0.04
                        
                        def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
                            p_ret = np.sum(mean_returns * weights)
                            p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                            return -((p_ret - risk_free_rate) / p_vol)
                            
                        num_assets = len(t_list)
                        args = (mean_returns, cov_matrix, risk_free_rate)
                        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
                        initial_guess = num_assets * [1. / num_assets,]
                        
                        optimized = sco.minimize(negative_sharpe, initial_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
                        
                        if optimized.success:
                            opt_weights = optimized.x
                            p_ret = np.sum(mean_returns * opt_weights)
                            p_vol = np.sqrt(np.dot(opt_weights.T, np.dot(cov_matrix, opt_weights)))
                            p_sharpe = (p_ret - risk_free_rate) / p_vol
                            st.session_state.port_results = {'weights': opt_weights, 'ret': p_ret, 'vol': p_vol, 'sharpe': p_sharpe, 'tickers': list(port_data.columns)}
                        else:
                            st.error("Optimization Failed.")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        if st.session_state.port_results is not None:
            res = st.session_state.port_results
            st.success("✅ **Optimization Successful!**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Expected Annual Return", f"{res['ret']*100:.2f}%")
            m2.metric("Portfolio Volatility (Risk)", f"{res['vol']*100:.2f}%")
            m3.metric("Sharpe Ratio", f"{res['sharpe']:.2f}")
            
            fig_pie = go.Figure(data=[go.Pie(labels=res['tickers'], values=res['weights'], hole=.4)])
            st.plotly_chart(fig_pie, use_container_width=True)

    # ==========================================
    # TAB 6: GUIDE
    # ==========================================
    with tab6:
        st.markdown("### 📖 Terminal Operations Guide")
        st.markdown("""
        **1. Technical Analysis (Top Chart):** Use the sidebar to toggle SMA (trend direction) and RSI (overbought/oversold levels). <br>
        **2. Statistical Models (Tab 1):** Tune p, d, and q sliders to test ARIMA and SARIMA combinations. Look for the lowest RMSE. <br>
        **3. ML Models (Tab 2):** Great for baseline, calendar-based trends using Prophet or Random Forest. <br>
        **4. Macro SARIMAX (Tab 3):** The ultimate forecasting tool. Download your CSV/PDFs here! <br>
        **5. Macro Scanner (Tab 4):** Run the heatmap to find which global driver to use in Tab 3. <br>
        **6. Portfolio Optimizer (Tab 5):** Input a mix of assets to find the exact percentage to hold of each to maximize return for the lowest mathematical risk.
        """)
