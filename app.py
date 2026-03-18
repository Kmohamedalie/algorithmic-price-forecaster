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
raw_ticker_input = st.sidebar.text_input("Target Ticker (For Tabs 1-3)", "AAPL").upper()
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
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Forecasting Models", "🌍 Macro Scanner", "⚖️ Portfolio Optimizer", "📖 Strategy Guide"])

    # ==========================================
    # TAB 1: FORECASTING (Combined Stats & ML for neatness)
    # ==========================================
    with tab1:
        st.subheader("Time-Series Forecasting")
        model_choice = st.selectbox("Choose Engine", ["Prophet (Algorithmic)", "ARIMA (Statistical)", "Random Forest (ML)"])
        
        if st.button("Run Forecast Engine"):
            with st.spinner(f"Running {model_choice}..."):
                try:
                    last_date = df['Date'].iloc[-1]
                    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
                    df_train = df['Close'].dropna().values
                    
                    if model_choice == "Prophet (Algorithmic)":
                        df_p = df[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
                        m = Prophet()
                        m.fit(df_p)
                        future = m.make_future_dataframe(periods=days_to_predict)
                        forecast = m.predict(future)['yhat'].tail(days_to_predict).values
                        rmse = np.sqrt(mean_squared_error(df_p['y'], m.predict(df_p)['yhat']))
                        
                    elif model_choice == "ARIMA (Statistical)":
                        model = ARIMA(df_train, order=(5,1,0))
                        fitted = model.fit()
                        forecast = fitted.forecast(steps=days_to_predict)
                        rmse = np.sqrt(mean_squared_error(df_train[30:], fitted.fittedvalues[30:]))
                        
                    elif model_choice == "Random Forest (ML)":
                        df_rf = df[['Date', 'Close']].dropna().copy()
                        df_rf['Day'] = df_rf['Date'].dt.day
                        df_rf['Month'] = df_rf['Date'].dt.month
                        df_rf['Year'] = df_rf['Date'].dt.year
                        X = df_rf[['Day', 'Month', 'Year']]
                        y = df_rf['Close']
                        rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
                        
                        future_df = pd.DataFrame({'Date': future_dates})
                        future_df['Day'] = future_df['Date'].dt.day
                        future_df['Month'] = future_df['Date'].dt.month
                        future_df['Year'] = future_df['Date'].dt.year
                        forecast = rf.predict(future_df[['Day', 'Month', 'Year']])
                        rmse = np.sqrt(mean_squared_error(y, rf.predict(X)))

                    st.metric(label=f"Historical RMSE Grade", value=round(rmse, 4))
                    
                    fig_fc = go.Figure()
                    fig_fc.add_trace(go.Scatter(x=df['Date'].tail(150), y=df['Close'].tail(150), name="Actual"))
                    fig_fc.add_trace(go.Scatter(x=future_dates, y=forecast, name="Predicted", line=dict(dash='dot', width=3)))
                    st.plotly_chart(fig_fc, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")

    # ==========================================
    # TAB 2: MACRO SCANNER
    # ==========================================
    with tab2:
        st.subheader("Global Macroeconomic Correlation")
        st.markdown("Scan the market to see what outside forces are driving your target asset right now.")
        
        if st.button("Generate Correlation Heatmap"):
            with st.spinner("Scanning Global Markets..."):
                # Download main drivers
                assets = {
                    "Target Asset": ticker, 
                    "US Dollar": "DX-Y.NYB", 
                    "Gold": "GC=F", 
                    "Crude Oil": "CL=F", 
                    "10Y Yield": "^TNX", 
                    "S&P 500": "^GSPC"
                }
                corr_df = pd.DataFrame()
                
                for name, symbol in assets.items():
                    tmp_df = yf.download(symbol, start=start_date, end=end_date, progress=False)
                    if not tmp_df.empty and 'Close' in tmp_df:
                        # Handle MultiIndex column drop for yfinance
                        close_col = tmp_df['Close'].iloc[:, 0] if isinstance(tmp_df['Close'], pd.DataFrame) else tmp_df['Close']
                        corr_df[name] = close_col
                        
                corrs = corr_df.corr()
                fig_corr = go.Figure(data=go.Heatmap(z=corrs.values, x=corrs.columns, y=corrs.columns, colorscale='RdBu', zmin=-1, zmax=1))
                fig_corr.update_layout(title="Asset Correlation Matrix (-1 to +1)")
                
                st.session_state.corr_data = fig_corr
                
        if st.session_state.corr_data:
            st.plotly_chart(st.session_state.corr_data, use_container_width=True)
            st.info("💡 **How to read this:** Look at the 'Target Asset' row. Dark Blue (+1) means they move perfectly together. Dark Red (-1) means they move in opposite directions. Use the darkest colored asset as your exogenous variable in a SARIMAX model!")

    # ==========================================
    # TAB 3: PORTFOLIO OPTIMIZER (Quadratic Math)
    # ==========================================
    with tab3:
        st.subheader("⚖️ Modern Portfolio Theory (MPT)")
        st.markdown("Enter multiple tickers. The algorithm will simulate thousands of combinations to find the allocation that maximizes your returns while minimizing your risk (Max Sharpe Ratio).")
        
        multi_tickers = st.text_input("Portfolio Tickers (comma separated)", "AAPL, MSFT, GLD, BTC-USD").upper()
        
        if st.button("Optimize Weights"):
            with st.spinner("Calculating the Efficient Frontier..."):
                t_list = [x.strip() for x in multi_tickers.split(",")]
                
                if len(t_list) < 2:
                    st.error("You need at least 2 assets to build a portfolio!")
                else:
                    try:
                        # 1. Fetch data
                        port_data = yf.download(t_list, start=start_date, end=end_date)['Close']
                        port_data.dropna(inplace=True)
                        
                        # 2. Calculate Returns & Covariance
                        returns = port_data.pct_change().dropna()
                        mean_returns = returns.mean() * 252  # Annualized
                        cov_matrix = returns.cov() * 252     # Annualized
                        risk_free_rate = 0.04                # 4% safe yield
                        
                        # 3. Optimization Functions
                        def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate):
                            p_ret = np.sum(mean_returns * weights)
                            p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                            p_sharpe = (p_ret - risk_free_rate) / p_vol
                            return p_ret, p_vol, p_sharpe
                            
                        def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
                            p_ret, p_vol, p_sharpe = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
                            return -p_sharpe # Scipy minimizes, so we minimize the negative Sharpe
                            
                        # 4. Scipy Constraints & Bounds
                        num_assets = len(t_list)
                        args = (mean_returns, cov_matrix, risk_free_rate)
                        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # Weights must = 100%
                        bounds = tuple((0.0, 1.0) for asset in range(num_assets))      # No short selling
                        initial_guess = num_assets * [1. / num_assets,]                # Start with equal weight
                        
                        # 5. Execute SLSQP Optimizer
                        optimized = sco.minimize(negative_sharpe, initial_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
                        
                        if optimized.success:
                            opt_weights = optimized.x
                            p_ret, p_vol, p_sharpe = portfolio_performance(opt_weights, mean_returns, cov_matrix, risk_free_rate)
                            
                            # 6. Save to session state
                            st.session_state.port_results = {
                                'weights': opt_weights, 'ret': p_ret, 'vol': p_vol, 'sharpe': p_sharpe, 'tickers': list(port_data.columns)
                            }
                        else:
                            st.error("Mathematical Optimization Failed. Try different dates or assets.")
                            
                    except Exception as e:
                        st.error(f"Error fetching data or calculating matrices: {e}")
        
        # Display the locked session state
        if st.session_state.port_results is not None:
            res = st.session_state.port_results
            
            st.success("✅ **Optimization Successful!** Here is your mathematically perfect portfolio:")
            
            # Key Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Expected Annual Return", f"{res['ret']*100:.2f}%")
            m2.metric("Portfolio Volatility (Risk)", f"{res['vol']*100:.2f}%")
            m3.metric("Sharpe Ratio", f"{res['sharpe']:.2f}")
            
            # Allocation Display
            st.markdown("#### Optimal Capital Allocation")
            weight_cols = st.columns(len(res['tickers']))
            for i, t in enumerate(res['tickers']):
                weight_cols[i].metric(t, f"{res['weights'][i]*100:.1f}%")
                
            # Plotly Donut Chart
            fig_pie = go.Figure(data=[go.Pie(labels=res['tickers'], values=res['weights'], hole=.4, hoverinfo="label+percent")])
            fig_pie.update_layout(title_text="Max Sharpe Ratio Portfolio Weights")
            st.plotly_chart(fig_pie, use_container_width=True)

    # ==========================================
    # TAB 4: GUIDE
    # ==========================================
    with tab4:
        st.markdown("### 📖 Terminal Operations Guide")
        st.markdown("""
        **1. Technical Analysis (Top Chart):** Use the sidebar to toggle SMA (trend direction) and RSI (overbought/oversold levels).
        
        **2. Forecasting Models (Tab 1):** * **ARIMA:** Best for short-term statistical momentum.
        * **Random Forest:** Machine Learning model that looks for calendar-based repeating patterns.
        * **Prophet:** Best for long-term algorithmic trend identification.
        
        **3. Macro Scanner (Tab 2):** Run the heatmap to see what global drivers correlate with your asset. If your asset is dark red against the US Dollar, it means your asset crashes when the Dollar rises.
        
        **4. Portfolio Optimizer (Tab 3):** Input a mix of uncorrelated assets (e.g., a tech stock, gold, and crypto). The quadratic optimizer will find the exact percentage to hold of each to squeeze out the highest return for the lowest mathematical risk.
        """)
