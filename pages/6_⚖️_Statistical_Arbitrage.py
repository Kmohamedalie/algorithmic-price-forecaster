import streamlit as st
import yfinance as yf
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import plotly.graph_objects as go

st.set_page_config(page_title="Statistical Arbitrage", layout="wide")

st.title("⚖️ Statistical Arbitrage: Pairs Trading")
st.markdown("Tests two assets for mathematical cointegration and visualizes the Z-Score to find mean-reversion trading opportunities.")

# --- INPUTS ---
col1, col2, col3 = st.columns(3)
with col1:
    asset_1 = st.text_input("Asset 1 (e.g., CVX)", "CVX").upper() # Chevron
with col2:
    asset_2 = st.text_input("Asset 2 (e.g., XOM)", "XOM").upper() # ExxonMobil
with col3:
    window = st.number_input("Rolling Window (Days)", min_value=10, max_value=100, value=20)

if st.button("Run Cointegration & Z-Score Analysis"):
    with st.spinner(f"Analyzing relationship between {asset_1} and {asset_2}..."):
        
        # 1. Fetch Data
        df = yf.download([asset_1, asset_2], start="2022-01-01", end=pd.to_datetime("today"), progress=False)['Close']
        
        if df.empty or len(df.columns) < 2:
            st.error("Could not fetch data for both assets. Please check the tickers.")
        else:
            df = df.dropna()
            
            # 2. Cointegration Test (The Rubber Band Test)
            # Null hypothesis: The assets are NOT cointegrated. We want a p-value < 0.05 to reject this.
            score, p_value, _ = coint(df[asset_1], df[asset_2])
            
            # 3. Calculate the Spread and Z-Score
            # We use Ordinary Least Squares (OLS) regression to find the exact hedge ratio
            X = sm.add_constant(df[asset_2])
            model = sm.OLS(df[asset_1], X).fit()
            hedge_ratio = model.params.iloc[1]
            
            # The Spread = Asset 1 - (Hedge Ratio * Asset 2)
            df['Spread'] = df[asset_1] - (hedge_ratio * df[asset_2])
            
            # The Z-Score normalizes the spread.
            rolling_mean = df['Spread'].rolling(window=window).mean()
            rolling_std = df['Spread'].rolling(window=window).std()
            df['Z-Score'] = (df['Spread'] - rolling_mean) / rolling_std
            
            # --- UI RENDERING ---
            st.subheader(f"Step 1: The Cointegration Test")
            
            c1, c2 = st.columns(2)
            c1.metric("Calculated Hedge Ratio", f"{hedge_ratio:.3f}", help=f"For every 1 share of {asset_1}, you need to short {hedge_ratio:.3f} shares of {asset_2}.")
            
            if p_value < 0.05:
                c2.metric("P-Value", f"{p_value:.4f}", "🟢 Cointegrated (Safe to Trade)")
                st.success(f"**Mathematical Proof:** {asset_1} and {asset_2} are fundamentally linked. They will eventually snap back together.")
            else:
                c2.metric("P-Value", f"{p_value:.4f}", "🔴 NOT Cointegrated (Dangerous)")
                st.warning(f"**Warning:** These assets are NOT mathematically linked. Any divergence is likely permanent. Do not trade this pair.")

            st.markdown("---")
            
            # 4. Plot the Z-Score
            st.subheader("Step 2: The Z-Score Oscillator")
            st.markdown("When the Z-Score crosses **+2.0** or **-2.0**, the rubber band is severely stretched. This is the exact moment StatArb algorithms execute the trade.")
            
            fig = go.Figure()
            
            # Plot the actual Z-Score line
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df['Z-Score'], 
                mode='lines', 
                name='Z-Score',
                line=dict(color='blue', width=1.5)
            ))
            
            # Add the Overbought/Oversold (+2 and -2) Bands
            fig.add_hline(y=2.0, line_dash="dash", line_color="red", annotation_text="Short Asset 1, Buy Asset 2 (+2.0)")
            fig.add_hline(y=-2.0, line_dash="dash", line_color="green", annotation_text="Buy Asset 1, Short Asset 2 (-2.0)")
            fig.add_hline(y=0.0, line_dash="solid", line_color="black", annotation_text="Mean (Take Profit)")
            
            fig.update_layout(
                yaxis_title="Z-Score (Standard Deviations)",
                height=450,
                template="plotly_white",
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 5. Live Trading Signal
            latest_z = df['Z-Score'].iloc[-1]
            st.subheader("🤖 Current Live Algorithm Signal")
            if latest_z >= 2.0:
                st.error(f"🚨 **EXECUTE SHORT SPREAD:** Z-Score is {latest_z:.2f}. Short {asset_1} and Buy {asset_2}.")
            elif latest_z <= -2.0:
                st.success(f"🚨 **EXECUTE LONG SPREAD:** Z-Score is {latest_z:.2f}. Buy {asset_1} and Short {asset_2}.")
            else:
                st.info(f"💤 **HOLD/WAIT:** Z-Score is {latest_z:.2f}. The rubber band is not stretched enough to trade right now.")
