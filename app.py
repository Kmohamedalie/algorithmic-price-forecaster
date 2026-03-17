"""
Streamlit Quantitative Stock Price Forecasting Application

Three forecasting approaches:
  1. ARIMA  - Statistical Analysis
  2. Random Forest Regressor - Machine Learning
  3. Facebook Prophet - Additive Time-Series Modeling

Disclaimer: For educational / data-science demonstration purposes only.
"""

import warnings
import logging
import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Stock Price Forecasting",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_stock_data(ticker: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance and return a clean DataFrame."""
    df = yf.download(ticker, start=str(start), end=str(end), auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df


def forecast_arima(series: pd.Series, horizon: int) -> pd.Series:
    """Fit an ARIMA(5,1,0) model and return a forecast series."""
    model = ARIMA(series, order=(5, 1, 0))
    result = model.fit()
    forecast = result.forecast(steps=horizon)
    last_date = series.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.offsets.BDay(1), periods=horizon)
    return pd.Series(forecast.values, index=future_dates, name="ARIMA Forecast")


def forecast_random_forest(series: pd.Series, horizon: int, n_lags: int = 20) -> pd.Series:
    """
    Build lag features from the price series, train a Random Forest Regressor,
    and iteratively predict `horizon` future trading days.
    """
    scaler = MinMaxScaler()
    values = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()

    def make_features(arr: np.ndarray, lags: int):
        X, y = [], []
        for i in range(lags, len(arr)):
            X.append(arr[i - lags: i])
            y.append(arr[i])
        return np.array(X), np.array(y)

    X, y = make_features(values, n_lags)
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, y)

    # Iterative multi-step forecast
    last_window = values[-n_lags:].tolist()
    preds_scaled = []
    for _ in range(horizon):
        pred = model.predict(np.array(last_window[-n_lags:]).reshape(1, -1))[0]
        preds_scaled.append(pred)
        last_window.append(pred)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    last_date = series.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.offsets.BDay(1), periods=horizon)
    return pd.Series(preds, index=future_dates, name="RF Forecast")


def forecast_prophet(series: pd.Series, horizon: int) -> pd.Series:
    """Fit a Facebook Prophet model and return a forecast series."""
    from prophet import Prophet  # imported here to keep startup fast

    df_prophet = pd.DataFrame({"ds": series.index, "y": series.values})
    df_prophet["ds"] = df_prophet["ds"].dt.tz_localize(None)
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=horizon, freq="B")
    forecast = model.predict(future)
    forecast_tail = forecast.set_index("ds")[["yhat"]].tail(horizon)
    forecast_tail.index = pd.to_datetime(forecast_tail.index)
    return pd.Series(forecast_tail["yhat"].values, index=forecast_tail.index, name="Prophet Forecast")


def build_forecast_chart(
    historical: pd.Series,
    forecast: pd.Series,
    title: str,
    hist_color: str = "#636EFA",
    fore_color: str = "#EF553B",
) -> go.Figure:
    """Return a Plotly figure with historical prices and the forecast overlay."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=historical.index,
            y=historical.values,
            mode="lines",
            name="Historical",
            line=dict(color=hist_color, width=2),
        )
    )
    # Connect last historical point to first forecast point
    connect_x = [historical.index[-1]] + list(forecast.index)
    connect_y = [historical.iloc[-1]] + list(forecast.values)
    fig.add_trace(
        go.Scatter(
            x=connect_x,
            y=connect_y,
            mode="lines",
            name="Forecast",
            line=dict(color=fore_color, width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Close Price (USD)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
        height=380,
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar – user inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuration")

    ticker = st.text_input(
        "Stock Ticker Symbol",
        value="AAPL",
        help="Enter a valid Yahoo Finance ticker (e.g. AAPL, MSFT, TSLA).",
    ).upper().strip()

    today = datetime.date.today()
    default_start = today - datetime.timedelta(days=5 * 365)

    start_date = st.date_input("Start Date", value=default_start, max_value=today)
    end_date = st.date_input("End Date", value=today, max_value=today)

    forecast_horizon = st.slider(
        "Forecast Horizon (trading days)",
        min_value=5,
        max_value=90,
        value=30,
        step=5,
    )

    run_btn = st.button("🚀 Run Forecast", use_container_width=True)

    st.markdown("---")
    st.markdown(
        "**⚠️ Disclaimer**\n\n"
        "This application is built **strictly for educational and data-science "
        "demonstration purposes**. Financial markets are highly volatile, and no "
        "algorithm can predict stock prices with 100% certainty. "
        "**Do not use these models for real-world financial trading.**"
    )

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
st.title("📈 Stock Price Forecasting")
st.markdown(
    "Compare three distinct forecasting approaches — **ARIMA**, "
    "**Random Forest Regressor**, and **Facebook Prophet** — applied to live "
    "stock market data fetched via [yfinance](https://github.com/ranaroussi/yfinance)."
)

if not run_btn:
    st.info("Configure the parameters in the sidebar and click **🚀 Run Forecast** to begin.")
    st.stop()

# Validate inputs
if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

# ---------------------------------------------------------------------------
# Fetch data
# ---------------------------------------------------------------------------
with st.spinner(f"Fetching data for **{ticker}** …"):
    try:
        df = fetch_stock_data(ticker, start_date, end_date)
    except Exception as exc:
        st.error(f"Failed to fetch data: {exc}")
        st.stop()

if df.empty:
    st.error(
        f"No price data found for **{ticker}** between {start_date} and {end_date}. "
        "Please verify the ticker symbol and date range."
    )
    st.stop()

series = df["Close"]

st.success(
    f"✅ Loaded **{len(series):,}** trading days of data for **{ticker}** "
    f"({series.index[0].date()} → {series.index[-1].date()})."
)

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest Close", f"${series.iloc[-1]:.2f}")
col2.metric("52-Week High", f"${series[-252:].max():.2f}")
col3.metric("52-Week Low", f"${series[-252:].min():.2f}")
price_chg = (series.iloc[-1] - series.iloc[-2]) / series.iloc[-2] * 100
col4.metric("1-Day Change", f"{price_chg:+.2f}%", delta_color="normal")

st.markdown("---")

# ---------------------------------------------------------------------------
# Run forecasts
# ---------------------------------------------------------------------------
tabs = st.tabs([
    "📊 ARIMA",
    "🌲 Random Forest",
    "🔮 Prophet",
    "📋 Data Table",
])

errors: dict[str, str] = {}
forecasts: dict[str, pd.Series] = {}

# ── ARIMA ────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("ARIMA – Statistical Time-Series Forecast")
    st.markdown(
        "**AutoRegressive Integrated Moving Average (ARIMA)** models a time series "
        "as a linear combination of its own past values and past forecast errors. "
        "Here we use an ARIMA(5, 1, 0) specification — 5 autoregressive lags with "
        "first-order differencing to achieve stationarity."
    )
    with st.spinner("Fitting ARIMA model …"):
        try:
            arima_fc = forecast_arima(series, forecast_horizon)
            forecasts["ARIMA"] = arima_fc
            fig_arima = build_forecast_chart(
                series,
                arima_fc,
                f"{ticker} – ARIMA({forecast_horizon}d Forecast)",
                hist_color="#636EFA",
                fore_color="#EF553B",
            )
            st.plotly_chart(fig_arima, use_container_width=True)
        except Exception as exc:
            errors["ARIMA"] = str(exc)
            st.error(f"ARIMA failed: {exc}")

# ── Random Forest ─────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Random Forest Regressor – Machine Learning Forecast")
    st.markdown(
        "A **Random Forest Regressor** is trained on lag features derived from the "
        "closing-price series (20 lagged values). Predictions are generated "
        "iteratively: each newly predicted value is appended to the feature window "
        "to predict the next step."
    )
    with st.spinner("Training Random Forest model …"):
        try:
            rf_fc = forecast_random_forest(series, forecast_horizon)
            forecasts["Random Forest"] = rf_fc
            fig_rf = build_forecast_chart(
                series,
                rf_fc,
                f"{ticker} – Random Forest ({forecast_horizon}d Forecast)",
                hist_color="#636EFA",
                fore_color="#00CC96",
            )
            st.plotly_chart(fig_rf, use_container_width=True)
        except Exception as exc:
            errors["Random Forest"] = str(exc)
            st.error(f"Random Forest failed: {exc}")

# ── Prophet ───────────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Facebook Prophet – Additive Time-Series Forecast")
    st.markdown(
        "**Prophet** decomposes the time series into trend, weekly seasonality, and "
        "yearly seasonality components using an additive model. It is robust to "
        "missing data and outliers and automatically handles holiday effects."
    )
    with st.spinner("Fitting Prophet model …"):
        try:
            prophet_fc = forecast_prophet(series, forecast_horizon)
            forecasts["Prophet"] = prophet_fc
            fig_prophet = build_forecast_chart(
                series,
                prophet_fc,
                f"{ticker} – Prophet ({forecast_horizon}d Forecast)",
                hist_color="#636EFA",
                fore_color="#AB63FA",
            )
            st.plotly_chart(fig_prophet, use_container_width=True)
        except Exception as exc:
            errors["Prophet"] = str(exc)
            st.error(f"Prophet failed: {exc}")

# ── Data Table ────────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Forecast Values – Raw Data")
    if forecasts:
        fc_df = pd.DataFrame(forecasts)
        fc_df.index.name = "Date"
        fc_df = fc_df.rename_axis("Date").reset_index()
        fc_df["Date"] = fc_df["Date"].dt.date
        st.dataframe(fc_df.style.format({c: "${:.2f}" for c in fc_df.columns if c != "Date"}),
                     use_container_width=True)
        csv = fc_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download CSV", csv, f"{ticker}_forecast.csv", "text/csv")
    else:
        st.warning("No forecasts available to display.")

# ---------------------------------------------------------------------------
# Combined comparison chart
# ---------------------------------------------------------------------------
if len(forecasts) > 1:
    st.markdown("---")
    st.subheader("📊 All-Model Comparison")

    colors = {"ARIMA": "#EF553B", "Random Forest": "#00CC96", "Prophet": "#AB63FA"}
    fig_cmp = go.Figure()
    fig_cmp.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name="Historical",
            line=dict(color="#636EFA", width=2),
        )
    )
    for name, fc in forecasts.items():
        connect_x = [series.index[-1]] + list(fc.index)
        connect_y = [series.iloc[-1]] + list(fc.values)
        fig_cmp.add_trace(
            go.Scatter(
                x=connect_x,
                y=connect_y,
                mode="lines",
                name=name,
                line=dict(color=colors.get(name, "#888"), width=2, dash="dash"),
            )
        )
    fig_cmp.update_layout(
        title=f"{ticker} – Model Comparison ({forecast_horizon}d Forecast)",
        xaxis_title="Date",
        yaxis_title="Close Price (USD)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
        height=450,
    )
    st.plotly_chart(fig_cmp, use_container_width=True)
