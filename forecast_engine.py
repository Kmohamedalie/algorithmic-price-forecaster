import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings

# Ignore statistical convergence warnings in the console
warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION (Edit these manually)
# ==========================================
TARGET_TICKER = "AAPL"          # Try "BTC-USD", "EURUSD=X", etc.
MACRO_TICKER = "DX-Y.NYB"       # The exogenous variable (US Dollar Index)
DAYS_TO_PREDICT = 14

# Calculate dates (Past 3 years to today)
END_DATE = date.today()
START_DATE = END_DATE - timedelta(days=365*3)

# ==========================================
# 2. DATA FETCHING FUNCTION
# ==========================================
def load_data(ticker_symbol, start, end):
    print(f"Fetching data for {ticker_symbol}...")
    start_str = start.strftime('%Y-%m-%d')
    end_str = end.strftime('%Y-%m-%d')
    data = yf.download(ticker_symbol, start=start_str, end=end_str, progress=False)
    
    if not data.empty and isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    data.reset_index(inplace=True)
    return data

# Fetch main target data
df = load_data(TARGET_TICKER, START_DATE, END_DATE)
df_train_values = df['Close'].dropna().values
skip_days = 30 if len(df_train_values) > 60 else 0 

print(f"\n--- Running Models for {TARGET_TICKER} ---")

# ==========================================
# 3. STATISTICAL MODEL (ARIMA)
# ==========================================
print("\n[1] Running Classic ARIMA...")
p, d, q = 5, 1, 0  # Standard ARIMA parameters

try:
    arima_model = ARIMA(df_train_values, order=(p, d, q))
    arima_fitted = arima_model.fit()
    arima_forecast = arima_fitted.forecast(steps=DAYS_TO_PREDICT)
    
    arima_rmse = np.sqrt(mean_squared_error(df_train_values[skip_days:], arima_fitted.fittedvalues[skip_days:]))
    print(f"ARIMA RMSE Grade: {arima_rmse:.4f}")
except Exception as e:
    print(f"ARIMA failed: {e}")

# ==========================================
# 4. ALGORITHMIC MODEL (Prophet)
# ==========================================
print("\n[2] Running Facebook Prophet...")
try:
    df_prophet = df[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
    m = Prophet(daily_seasonality=True)
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=DAYS_TO_PREDICT)
    prophet_forecast_full = m.predict(future)
    
    prophet_historical = prophet_forecast_full['yhat'][:len(df_prophet)]
    prophet_rmse = np.sqrt(mean_squared_error(df_prophet['y'], prophet_historical))
    print(f"Prophet RMSE Grade: {prophet_rmse:.4f}")
except Exception as e:
    print(f"Prophet failed: {e}")

# ==========================================
# 5. MACROECONOMIC SARIMAX (With Exogenous)
# ==========================================
print(f"\n[3] Running Macro SARIMAX (using {MACRO_TICKER} as outside driver)...")
mp, md, mq = 0, 1, 0  # Optimal parameters based on earlier testing

macro_df = load_data(MACRO_TICKER, START_DATE, END_DATE)

if macro_df.empty:
    print(f"Failed to fetch macroeconomic data for {MACRO_TICKER}.")
else:
    # Merge datasets on Date to ensure alignment
    merged_df = pd.merge(df[['Date', 'Close']], macro_df[['Date', 'Close']], on='Date', suffixes=('_Target', '_Macro'))
    merged_df.dropna(inplace=True)

    endog = merged_df['Close_Target'].values
    exog = merged_df['Close_Macro'].values

    try:
        macro_model = SARIMAX(endog, exog=exog, order=(mp, md, mq))
        macro_fitted = macro_model.fit(disp=False)
        
        # Naive forecast for the exogenous variable (assuming it stays flat)
        future_exog = np.repeat(exog[-1], DAYS_TO_PREDICT)
        macro_forecast = macro_fitted.forecast(steps=DAYS_TO_PREDICT, exog=future_exog)
        
        # Calculate Macro RMSE
        macro_skip = 30 if len(endog) > 60 else 0
        macro_rmse = np.sqrt(mean_squared_error(endog[macro_skip:], macro_fitted.fittedvalues[macro_skip:]))
        print(f"Macro SARIMAX RMSE Grade: {macro_rmse:.4f}")
        
        # --- DATA EXPORT ---
        last_date = merged_df['Date'].iloc[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, DAYS_TO_PREDICT + 1)]
        
        output_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Target_Price': macro_forecast,
            'Assumed_Exogenous_Value': future_exog
        })
        
        # Save directly to computer
        filename = f"{TARGET_TICKER}_Legacy_Forecast.csv"
        output_df.to_csv(filename, index=False)
        print(f"\n✅ SUCCESS: Forecast data saved to '{filename}' in your current directory.")
        
    except Exception as e:
        print(f"Macro SARIMAX failed: {e}")
