import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime

# Download the VADER dictionary (runs silently in the background)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

st.set_page_config(page_title="NLP Sentiment Analysis", layout="wide")

st.title("📰 NLP Market Sentiment Scanner")
st.markdown("Uses Natural Language Processing (VADER Lexicon) to read live news headlines and mathematically score market emotion.")

# --- INPUT ---
ticker = st.text_input("Enter Ticker to Scan News", "AAPL").upper()

if st.button("Run NLP Scanner"):
    with st.spinner(f"Scraping live news and running NLP analysis for {ticker}..."):
        
        # 1. Initialize the NLP Brain
        sia = SentimentIntensityAnalyzer()
        
        # 2. Fetch Live News via yfinance
        stock = yf.Ticker(ticker)
        news_data = stock.news
        
        if not news_data:
            st.error("No recent news found for this ticker.")
        else:
            analyzed_news = []
            total_score = 0
            
            # 3. Read and Score Each Headline
            for article in news_data:
                # yfinance news dict structures can vary, we safely grab the title
                title = article.get('title', '')
                publisher = article.get('publisher', 'Unknown')
                link = article.get('link', '#')
                
                # Timestamp conversion (yfinance provides unix timestamps)
                timestamp = article.get('providerPublishTime', 0)
                date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M') if timestamp else "Unknown Time"
                
                if title:
                    # The Magic NLP Line: Scores from -1 (Extremely Negative) to +1 (Extremely Positive)
                    sentiment = sia.polarity_scores(title)
                    compound_score = sentiment['compound']
                    total_score += compound_score
                    
                    # Human-readable label
                    if compound_score >= 0.05:
                        label = "🟢 Bullish"
                    elif compound_score <= -0.05:
                        label = "🔴 Bearish"
                    else:
                        label = "⚪ Neutral"
                        
                    analyzed_news.append({
                        "Date": date_str,
                        "Publisher": publisher,
                        "Headline": title,
                        "Score": compound_score,
                        "Sentiment": label,
                        "Link": link
                    })
            
            # 4. Calculate the Macro Sentiment
            avg_sentiment = total_score / len(analyzed_news) if analyzed_news else 0
            
            # --- UI RENDERING ---
            st.subheader(f"Macro Sentiment Score: {ticker}")
            
            # Create a professional Gauge Chart
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = avg_sentiment,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Average NLP Compound Score"},
                gauge = {
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': "black"},
                    'steps': [
                        {'range': [-1, -0.05], 'color': "rgba(255, 0, 0, 0.3)"},   # Bearish Zone
                        {'range': [-0.05, 0.05], 'color': "rgba(200, 200, 200, 0.3)"}, # Neutral Zone
                        {'range': [0.05, 1], 'color': "rgba(0, 255, 0, 0.3)"}      # Bullish Zone
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': avg_sentiment
                    }
                }
            ))
            
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
            
            # Display the raw data
            st.markdown("### 🧠 The Algorithm's Inner Monologue")
            st.markdown("Here is exactly how the NLP engine scored the latest headlines:")
            
            # Convert to DataFrame for a beautiful Streamlit table
            df_display = pd.DataFrame(analyzed_news)
            
            # DEFENSIVE FIX: Check if the dataframe is empty before trying to display it
            if not df_display.empty:
                st.dataframe(
                    df_display[['Date', 'Publisher', 'Sentiment', 'Score', 'Headline']],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("The Yahoo Finance API did not return any readable headlines for this ticker at this time. Try a highly active ticker like 'TSLA', 'NVDA', or 'AAPL'.")
