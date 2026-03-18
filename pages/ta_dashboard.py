import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta # Technical Analysis library

st.set_page_config(page_title="Technical Analysis Dashboard", layout="wide")

st.title("📈 Stock Technical Analysis Dashboard")
st.markdown("Analyze Simple Moving Averages (SMA) and the Relative Strength Index (RSI).")

# --- USER INPUTS ---
col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.text_input("Enter Stock Ticker", value="AAPL").upper()
with col2:
    start_date = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
with col3:
    end_date = st.date_input("End Date", value=pd.to_datetime("today"))

if st.button("Fetch Data & Analyze"):
    with st.spinner(f"Fetching data for {ticker}..."):
        # 1. Fetch Data
        df = yf.download(ticker, start=start_date, end=end_date)
        
        if df.empty:
            st.error("No data found. Please check the ticker symbol or date range.")
        else:
            # FIX 1: Flatten yfinance multi-index if it exists
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            # 2. Calculate Indicators using the 'ta' library
            # 50-Day and 200-Day SMA
            df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
            df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
            
            # 14-Day RSI
            df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)

            # 3. Create the Plotly Figure (2 Rows: Top for Price/SMA, Bottom for RSI)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, 
                                row_heights=[0.7, 0.3])

            # ROW 1: Candlestick Chart
            fig.add_trace(go.Candlestick(x=df.index,
                                         open=df['Open'], high=df['High'],
                                         low=df['Low'], close=df['Close'],
                                         name='Price'), row=1, col=1)
            
            # ROW 1: Add SMAs
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], 
                                     line=dict(color='blue', width=1.5), 
                                     name='50-Day SMA'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], 
                                     line=dict(color='orange', width=1.5), 
                                     name='200-Day SMA'), row=1, col=1)

            # ROW 2: RSI Chart
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], 
                                     line=dict(color='purple', width=1.5), 
                                     name='RSI (14)'), row=2, col=1)
            
            # ROW 2: Add Overbought/Oversold reference lines (70 and 30)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, 
                          annotation_text="Overbought (70)", annotation_position="top right")
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, 
                          annotation_text="Oversold (30)", annotation_position="bottom right")

            # Formatting
            fig.update_layout(
                title=f"{ticker} - Price, SMA, and RSI",
                yaxis_title="Stock Price (USD)",
                yaxis2_title="RSI",
                xaxis_rangeslider_visible=False,
                xaxis=dict(
                    rangebreaks=[
                        dict(bounds=["sat", "mon"]) # FIX 2: Hides weekend gaps
                    ]
                ),
                height=700,
                template="plotly_white"
            )
            
            # Update RSI y-axis limits to 0-100
            fig.update_yaxes(range=[0, 100], row=2, col=1)

            # Display the chart
            st.plotly_chart(fig, use_container_width=True)

            # 4. Display Current Stats Summary
            st.subheader(f"Current Technical Summary for {ticker}")
            latest = df.iloc[-1]
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Latest Close Price", f"${latest['Close']:.2f}")
            
            # Logic for RSI Status
            rsi_val = latest['RSI_14']
            if pd.isna(rsi_val):
                rsi_status = "Not enough data"
            elif rsi_val >= 70:
                rsi_status = "🔴 Overbought"
            elif rsi_val <= 30:
                rsi_status = "🟢 Oversold"
            else:
                rsi_status = "⚪ Neutral"
                
            c2.metric("Current RSI (14)", f"{rsi_val:.2f}", rsi_status)
            
            # Logic for SMA Trend
            sma50, sma200 = latest['SMA_50'], latest['SMA_200']
            if pd.isna(sma50) or pd.isna(sma200):
                trend_status = "Not enough data"
            elif sma50 > sma200:
                trend_status = "⬆️ Bullish (50 > 200)"
            else:
                trend_status = "⬇️ Bearish (50 < 200)"
                
            c3.metric("Macro Trend (SMA)", f"${sma50:.2f} (50-Day)", trend_status)
