import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="Live Order Book", layout="wide")

st.title("🐋 Live Level 2 Order Book Scanner")
st.markdown("Visualizes real-time market microstructure and cumulative liquidity 'walls' using public exchange data.")

# --- INPUTS ---
col1, col2 = st.columns(2)
with col1:
    symbol = st.text_input("Crypto Pair (Binance Format)", "BTCUSDT").upper()
with col2:
    depth_limit = st.selectbox("Order Book Depth (Levels)", [100, 500, 1000, 5000], index=1)

if st.button("Fetch Live Snapshot"):
    with st.spinner(f"Pinging Binance matching engine for {symbol} Level 2 data..."):
        
        # 1. Fetch Live Data from Binance Public API
        url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={depth_limit}"
        response = requests.get(url)
        
        if response.status_code != 200:
            st.error(f"Failed to fetch data. Ensure '{symbol}' is a valid Binance pair (e.g., ETHUSDT, SOLUSDT).")
        else:
            data = response.json()
            
            # 2. Parse the Bids (Buyers) and Asks (Sellers)
            # Binance returns lists of strings: ["Price", "Quantity"]
            bids = pd.DataFrame(data['bids'], columns=['Price', 'Quantity'], dtype=float)
            asks = pd.DataFrame(data['asks'], columns=['Price', 'Quantity'], dtype=float)
            
            # --- DEFENSIVE CHECK: Ensure the order book isn't completely empty ---
            if bids.empty or asks.empty:
                st.warning(f"The order book for {symbol} returned empty. This might be a dead or delisted coin.")
            else:
                # 3. Calculate Cumulative Size (The "Walls")
                # Bids go from highest price to lowest. Asks go from lowest price to highest.
                bids = bids.sort_values(by='Price', ascending=False)
                bids['Cumulative_Qty'] = bids['Quantity'].cumsum()
                bids['Notional_Value'] = bids['Quantity'] * bids['Price'] # 🐛 FIXED: Real dollar value of each order
                
                asks = asks.sort_values(by='Price', ascending=True)
                asks['Cumulative_Qty'] = asks['Quantity'].cumsum()
                asks['Notional_Value'] = asks['Quantity'] * asks['Price'] # 🐛 FIXED: Real dollar value of each order
                
                # 4. Calculate Order Book Imbalance (OBI)
                total_bid_vol = bids['Quantity'].sum()
                total_ask_vol = asks['Quantity'].sum()
                
                obi = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol) if (total_bid_vol + total_ask_vol) > 0 else 0
                
                # --- UI RENDERING ---
                st.subheader(f"Market Microstructure: {symbol}")
                
                # Display OBI Metric
                if obi > 0.10:
                    obi_status = "🟢 High Buy Pressure (Bids > Asks)"
                elif obi < -0.10:
                    obi_status = "🔴 High Sell Pressure (Asks > Bids)"
                else:
                    obi_status = "⚪ Balanced Book"
                    
                c1, c2, c3 = st.columns(3)
                c1.metric("Current Spread (Middle Price)", f"${(bids['Price'].iloc[0] + asks['Price'].iloc[0])/2:.2f}")
                c2.metric("Order Book Imbalance (OBI)", f"{obi:.3f}", obi_status)
                c3.metric("Total Depth Tracked", f"{depth_limit * 2} Limit Orders")

                st.markdown("---")
                
                # 5. Plotly Depth Chart
                st.subheader("Liquidity Depth Chart")
                st.markdown("Look for sudden vertical 'steps' in the filled areas. These are massive limit orders acting as price walls.")
                
                fig = go.Figure()

                # Plot Buy Wall (Green)
                fig.add_trace(go.Scatter(
                    x=bids['Price'], 
                    y=bids['Cumulative_Qty'], 
                    mode='lines', 
                    fill='tozeroy', 
                    name='Bids (Buy Wall)',
                    line=dict(color='rgba(0, 255, 0, 0.8)', width=2),
                    fillcolor='rgba(0, 255, 0, 0.3)'
                ))

                # Plot Sell Wall (Red)
                fig.add_trace(go.Scatter(
                    x=asks['Price'], 
                    y=asks['Cumulative_Qty'], 
                    mode='lines', 
                    fill='tozeroy', 
                    name='Asks (Sell Wall)',
                    line=dict(color='rgba(255, 0, 0, 0.8)', width=2),
                    fillcolor='rgba(255, 0, 0, 0.3)'
                ))

                fig.update_layout(
                    xaxis_title="Asset Price (USD)",
                    yaxis_title="Cumulative Asset Quantity",
                    height=500,
                    template="plotly_white",
                    hovermode="x unified",
                    # Zoom in slightly to ignore ridiculous outlier orders at $1 or $1,000,000
                    xaxis=dict(range=[bids['Price'].min(), asks['Price'].max()])
                )

                st.plotly_chart(fig, use_container_width=True)
                
                # 6. Top Whales Table
                st.subheader("🕵️‍♂️ Whale Tracker: Largest Individual Orders")
                
                bids['Type'] = 'Buy'
                asks['Type'] = 'Sell'
                
                # Combine, sort by largest single order quantity, and take top 10
                combined = pd.concat([bids, asks])
                whales = combined.sort_values(by='Quantity', ascending=False).head(10)
                
                st.dataframe(
                    whales[['Type', 'Price', 'Quantity', 'Notional_Value']].style.format({
                        'Price': '${:.2f}',
                        'Quantity': '{:.4f}',
                        'Notional_Value': '${:,.2f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
