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
            st.error(f"No recent news found for {ticker} via Yahoo Finance.")
        else:
            analyzed_news = []
            total_score = 0
            
            # 3. Read and Score Each Headline
            for article in news_data:
                # Fallbacks for Yahoo Finance's constantly changing API structure
                title = article.get('title', article.get('content', dict()).get('title', ''))
                link = article.get('link', '#')
                
                # --- 🕵️‍♂️ UPGRADED PUBLISHER EXTRACTION ---
                # Check multiple possible dictionary keys Yahoo might be using today
                publisher = article.get('publisher') or article.get('source')
                if not publisher and 'provider' in article and isinstance(article['provider'], dict):
                    publisher = article['provider'].get('displayName')
                if not publisher:
                    publisher = "Unknown"
                
                # --- ⏱️ UPGRADED TIMESTAMP EXTRACTION ---
                # Yahoo sometimes uses integers (UNIX), sometimes strings (ISO 8601)
                raw_time = article.get('providerPublishTime') or article.get('pubDate') or article.get('publishedAt')
                
                date_str = "Unknown Time"
                if raw_time:
                    try:
                        if isinstance(raw_time, (int, float)): # It's a UNIX timestamp
                            date_str = datetime.fromtimestamp(raw_time).strftime('%Y-%m-%d %H:%M')
                        elif isinstance(raw_time, str): # It's a text string like "2024-03-21T15:30:00Z"
                            date_str = raw_time[:16].replace("T", " ") 
                    except Exception:
                        pass # If all math fails, it defaults back to "Unknown Time"
                
                # Only analyze if we successfully grabbed a real string of text
                if isinstance(title, str) and len(title) > 5:
                    sentiment = sia.polarity_scores(title)
                    compound_score = sentiment['compound']
                    total_score += compound_score
                    
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
            
            # 🚨 THE HARD STOP: Did we actually parse any text?
            if len(analyzed_news) == 0:
                st.warning(f"Yahoo Finance returned data for {ticker}, but the text structure was unreadable by the scanner. Try another ticker like 'NVDA' or 'TSLA'.")
            else:
                # 4. Calculate the Macro Sentiment (Only runs if we have data!)
                avg_sentiment = total_score / len(analyzed_news)
                
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
                
                df_display = pd.DataFrame(analyzed_news)
                st.dataframe(
                    df_display[['Date', 'Publisher', 'Sentiment', 'Score', 'Headline']],
                    use_container_width=True,
                    hide_index=True
                )
            
            # DEFENSIVE FIX: Check if the dataframe is empty before trying to display it
            if not df_display.empty:
                st.dataframe(
                    df_display[['Date', 'Publisher', 'Sentiment', 'Score', 'Headline']],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("The Yahoo Finance API did not return any readable headlines for this ticker at this time. Try a highly active ticker like 'TSLA', 'NVDA', or 'AAPL'.")
