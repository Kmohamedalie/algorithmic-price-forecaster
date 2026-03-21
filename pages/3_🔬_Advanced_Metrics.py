import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Advanced Quant Metrics", layout="wide")

st.title("🔬 Advanced Risk Analytics")
st.markdown("Go beyond the Sharpe Ratio. Analyze Downside Deviation (Sortino) and Maximum Drawdown (Calmar).")

# --- INPUTS ---
col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.text_input("Target Ticker", "QQQ").upper()
with col2:
    start_date = st.date_input("Start Date", pd.to_datetime("2018-01-01"))
with col3:
    risk_free_rate = st.number_input("Risk-Free Rate (%)", value=4.0, step=0.1) / 100

if st.button("Calculate Advanced Metrics"):
    with st.spinner(f"Crunching risk data for {ticker}..."):
        
        # 1. Fetch Data
        df = yf.download(ticker, start=start_date, end=pd.to_datetime("today"), progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if df.empty:
            st.error("No data found.")
        else:
            # 2. Base Calculations
            df['Return'] = df['Close'].pct_change()
            df = df.dropna()
            
            # Annualized Return & Volatility
            ann_return = df['Return'].mean() * 252
            ann_vol = df['Return'].std() * np.sqrt(252)
            
            # 3. ADVANCED METRIC: Sortino Ratio
            # Isolate only the negative returns
            downside_returns = df[df['Return'] < 0]['Return']
            downside_vol = downside_returns.std() * np.sqrt(252)
            
            # 4. ADVANCED METRIC: Calmar Ratio & Drawdown
            df['Cumulative'] = (1 + df['Return']).cumprod()
            df['High_Water_Mark'] = df['Cumulative'].cummax()
            df['Drawdown'] = (df['Cumulative'] - df['High_Water_Mark']) / df['High_Water_Mark']
            
            max_drawdown = abs(df['Drawdown'].min())
            
            # Calculate the Ratios
            sharpe_ratio = (ann_return - risk_free_rate) / ann_vol if ann_vol != 0 else 0
            sortino_ratio = (ann_return - risk_free_rate) / downside_vol if downside_vol != 0 else 0
            calmar_ratio = ann_return / max_drawdown if max_drawdown != 0 else 0

            # --- UI RENDERING ---
            st.subheader(f"Risk Profile: {ticker}")
            
            # Display Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Industry Standard: Sharpe Ratio", f"{sharpe_ratio:.2f}", help="Return vs. Total Volatility")
            m2.metric("Downside Focus: Sortino Ratio", f"{sortino_ratio:.2f}", help="Return vs. Downside Volatility only. Usually higher than Sharpe.")
            m3.metric("Crash Focus: Calmar Ratio", f"{calmar_ratio:.2f}", help="Return vs. Maximum Drawdown. > 1.0 is excellent.")
            
            st.markdown("---")
            
            # Visualizing the Pain: The Drawdown Chart
            st.subheader("Underwater Chart (Drawdown Profile)")
            st.markdown("This chart visualizes exactly how far the asset fell from its all-time highs on any given day. Institutional investors use this to gauge the 'psychological pain' of holding an asset.")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df['Drawdown'] * 100, 
                fill='tozeroy', 
                name="Drawdown %",
                line=dict(color='red', width=1),
                fillcolor='rgba(255, 0, 0, 0.3)'
            ))
            
            fig.update_layout(
                yaxis_title="Percentage Drop from Peak (%)",
                yaxis=dict(ticksuffix="%"),
                height=400,
                template="plotly_white",
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"💡 **Analysis:** The worst crash {ticker} experienced in this timeframe was a **{max_drawdown*100:.1f}%** drop.")
