# 🧮 The Mathematics of the Quant Terminal

This document outlines the statistical theories, mathematical formulas, and quantitative intuition powering the core analytical modules of the Multi-Asset Quantitative Strategy Terminal.


## The Main Dashboard: Technical Analysis Overlays
Before running complex forecasts, the terminal visualizes the current state of the asset using two fundamental quantitative indicators: Simple Moving Averages (SMA) and the Relative Strength Index (RSI).


### 1. Simple Moving Average (SMA)
The SMA strips away daily market noise (volatility) to reveal the underlying mathematical trend. It calculates the unweighted arithmetic mean of the asset's closing prices over a specified number of days ($n$).

**The Equation:**
$$SMA_n=\frac{1}{n}\sum_{i=0}^{n-1}P_{t-i}$$
*(Where P is the closing price at a given day, and n is the rolling window).*

*Intuition:* Our terminal plots the **50-day** (fast) and **200-day** (slow) SMAs. 
* When the current price is above the 200-day SMA, the asset is mathematically in a macro bull market. 
* When the 50-day SMA crosses *above* the 200-day SMA, it triggers a "Golden Cross," signaling a major upward momentum shift.

### 2. Relative Strength Index (RSI)

Developed by J. Welles Wilder, the RSI is a momentum oscillator that measures the speed and magnitude of recent price changes. It translates absolute price drops and gains into a normalized index scaled from 0 to 100.

**The Equations:**
First, we calculate the Relative Strength ($RS$):
$$RS=\frac{\text{Average Gain over } n \text{ periods}}{\text{Average Loss over } n \text{ periods}}$$
Then, we normalize it into the RSI:
$$RSI=100-\frac{100}{1+RS}$$

*Intuition:* The terminal calculates a standard 14-day RSI ($n=14$) and plots warning lines at the **70** and **30** levels.
* **RSI > 70 (Overbought):** The asset has gone up too fast, mathematically outpacing its historical average gains. A statistical pullback or correction is highly probable.
* **RSI < 30 (Oversold):** The asset has been heavily sold off, mathematically outpacing historical losses. It is statistically "cheap" and primed for a bounce.





## Tab 1: Classic Statistical Forecasting (ARIMA & ETS)
This tab relies on traditional time-series mathematics, which assumes that the future is a mathematical function of the past.

### 1. ARIMA (AutoRegressive Integrated Moving Average)
ARIMA predicts future prices by looking at its own past prices and past errors. It is defined by three parameters: (p, d, q).
* **AR(p) - AutoRegressive:** The model looks at *p* previous days. If *p* = 5, the model assumes today's price is a linear combination of the last 5 days.
* **I(d) - Integrated:** Financial data is usually "non-stationary" (it trends infinitely upward or downward). We difference the data *d* times (subtracting yesterday's price from today's) to make the math stable around a baseline of zero.
* **MA(q) - Moving Average:** The model looks at the error (the difference between its prediction and reality) of the last *q* days to smooth out sudden "shocks" in the market.

**The Equation:**
$$Y_t=c+\phi_1Y_{t-1}+\dots+\phi_pY_{t-p}+\theta_1\epsilon_{t-1}+\dots+\theta_q\epsilon_{t-q}+\epsilon_t$$
*(Where Y_t is the differenced price, phi represents past price weights, and theta represents past error weights).*

### 2. Exponential Smoothing (ETS)
Unlike a simple moving average that treats all past days equally, ETS gives exponentially decreasing weight to older observations. Yesterday is mathematically vastly more important than a day from three years ago.

---

## Tab 2: Algorithmic & Machine Learning
This tab steps away from classic linear statistics and uses modern, non-linear algorithmic approaches.

### 1. Facebook Prophet
Prophet does not look at the market as a sequence of steps; it looks at it as a curve made of overlapping waves. It uses a decomposable additive time-series model.

**The Equation:**
$$y(t)=g(t)+s(t)+h(t)+\epsilon_t$$
* **g(t):** The core trend (is the asset generally going up or down?).
* **s(t):** Seasonality (e.g., human behavior like "Crypto drops on Sundays").
* **h(t):** Holiday effects (anomalous days).

*Intuition:* Prophet is highly resilient to missing data (like weekends) and sudden trend shifts, making it exceptional for 24/7 assets like Bitcoin.

### 2. Random Forest Regressor
Instead of using time-series math, this uses an ensemble of Decision Trees. We extract "calendar features" (Day=15, Month=10, Year=2023) and ask the algorithm: *"Historically, when it is the middle of October, what does this asset usually do?"* It builds hundreds of hypothetical decision trees and averages their guesses together to prevent overfitting.

---

## Tab 3: Multivariate Macro SARIMAX
This is the holy grail of macroeconomic forecasting. SARIMAX takes the standard ARIMA model and introduces an **Exogenous Variable (X)**. 

Instead of just looking at Apple's past prices to predict Apple's future, SARIMAX mathematically factors in a completely separate, independent driver (like the US Dollar or Interest Rates). 

**The Equation:**
$$Y_t=\text{ARIMA}(p,d,q)+\beta X_t$$
*Intuition:* The beta * X_t term represents the outside macroeconomic force. If we use Crude Oil to predict an Airline stock, the model mathematically calculates how heavily a spike in oil prices will drag down the airline's future valuation.

---

## Tab 4: Macro Correlation Scanner
Before running SARIMAX, you must know *which* exogenous variable to use. This tab calculates the **Pearson Correlation Coefficient** for a matrix of global assets.

**The Equation:**
$$\rho_{X,Y}=\frac{\text{cov}(X,Y)}{\sigma_X\sigma_Y}$$
*Intuition:* The math measures the linear relationship between two assets. It outputs a score between **-1.0** and **+1.0**.
* If Asset A and Asset B have a correlation of **+0.90**, they move perfectly together.
* If they have a correlation of **-0.85**, they move in perfect opposite directions (e.g., when the US Dollar spikes, Gold crashes). The scanner visualizes this as a heatmap to allow the user to select the darkest-colored variable for their SARIMAX model.

---

## Tab 5: Modern Portfolio Theory (MPT) & The Optimizer

This is the most mathematically rigorous section of the terminal, utilizing **Quadratic Programming**. Pioneered by Harry Markowitz, MPT proves that by combining risky assets that are not perfectly correlated, you can mathematically reduce the overall risk of the portfolio without sacrificing returns.

### The Objective: Maximize the Sharpe Ratio
The algorithm's sole purpose is to find the exact allocation weights (*w*) of capital to maximize the Sharpe Ratio (return per unit of risk).

**The Sharpe Ratio Equation:**
$$S=\frac{R_p-R_f}{\sigma_p}$$
*(Where R_p is expected portfolio return, R_f is the risk-free rate, and sigma_p is portfolio volatility).*

### The Mathematical Components
To calculate the Sharpe Ratio, the algorithm must calculate two massive matrices for the portfolio:

**1. Expected Portfolio Return (R_p):** The weighted average of individual asset returns.
$$R_p=\sum_{i=1}^{n}w_i\mu_i$$

**2. Portfolio Variance / Risk:**
This is where the quadratic math happens. Risk is not just the average volatility of the assets; it must account for how the assets interact with each other (Covariance). 
$$\sigma_p^2=w^T\Sigma w$$
*(Where w is the vector of asset weights, w^T is its transpose, and Sigma is the covariance matrix of the assets).*

### The Constraints (SLSQP Algorithm)
We use the SciPy library to run Sequential Least Squares Programming. We ask the computer to test thousands of combinations to maximize the Sharpe Ratio, but we force it to obey two rules:
1.  Sum of weights = 1 (You must invest exactly 100% of your capital).
2.  Weights >= 0 (No short-selling; weights cannot be negative).

---

## Tab 6: Vectorized Algorithmic Backtesting
This module tests a systematic Quantitative Trading strategy (Moving Average Crossover) without using any loops, relying entirely on matrix vectorization for speed.

**The Core Logic:**
$$SMA_t=\frac{1}{n}\sum_{i=0}^{n-1}P_{t-i}$$
If SMA_fast > SMA_slow, the Signal = 1 (Buy). 

**The Lookahead Bias Prevention:**
In quantitative programming, you cannot trade today based on today's closing price, because you do not know the closing price until the market closes. Therefore, the code mathematically forces a 1-day delay:
$$Position_t=Signal_{t-1}$$
By shifting the signal vector forward, the backtester simulates real-world execution, ensuring the algorithm's reported ROI is historically accurate and not synthetically inflated.

## Tab 7: Advanced Risk Metrics (Sortino & Calmar)
While the Sharpe Ratio is the industry standard, it contains a critical mathematical flaw: Standard Deviation ($\sigma$) penalizes *all* volatility. If an asset suddenly explodes upward by 300%, the Sharpe Ratio drops because the asset became "volatile." Professional quants use advanced metrics to isolate only the "bad" volatility.

### 1. The Sortino Ratio (Downside Deviation)
The Sortino Ratio modifies the Sharpe equation by completely ignoring days where the portfolio made money. It only calculates the variance of the *negative* returns.

**The Equation:**
$$Sortino=\frac{R_p-R_f}{\sigma_d}$$
*(Where R_p is the portfolio return, R_f is the risk-free rate, and sigma_d is the Downside Deviation).*

*Intuition:* If an algorithm has a low Sharpe Ratio but a high Sortino Ratio, it tells the quant: *"This asset is highly volatile, but almost all of that volatility is the asset making massive, sudden profits."* 

### 2. Maximum Drawdown (MDD) & The Calmar Ratio

Standard deviation measures daily wiggles. Maximum Drawdown measures the absolute worst-case scenario: the percentage drop from the highest peak the portfolio ever reached to the lowest trough before a new peak is formed.

**The Equation:**
$$Calmar=\frac{R_p}{MDD}$$

*Intuition:* This is the ultimate test of psychological endurance. You could have an algorithm that returns 40% a year, but if it regularly suffers 60% drawdowns (MDD) along the way, most humans will panic and turn the bot off. The Calmar Ratio measures the return you get relative to the sheer terror of the worst crashes. A Calmar Ratio above 1.0 is generally considered elite.

---

## Tab 8: Natural Language Processing (NLP) & VADER
Quants no longer rely solely on numeric price data. Human emotion and news drive markets, which requires teaching algorithms to read unstructured text using Natural Language Processing.

### The VADER Lexicon
Our terminal uses VADER (Valence Aware Dictionary and sEntiment Reasoner). Unlike basic NLP models that just count "good" and "bad" words, VADER understands grammatical intensity. It mathematically weighs capitalization, punctuation, and modifiers. (e.g., "Profits drop" is negative, but "PROFITS PLUMMET!!!" is mathematically scored as severely negative).



### The Compound Score Normalization
When VADER reads a headline, it assigns a valence score (*v*) to every word. It sums these scores and normalizes them into a single `Compound Score` constrained between -1 (Extremely Bearish) and +1 (Extremely Bullish).

**The Equation:**
$$Compound=\frac{\sum v}{\sqrt{\sum v^2+\alpha}}$$
*(Where v is the valence score of each word, and alpha is a normalization constant, typically set to 15 to ensure the denominator is always slightly larger than the numerator, creating a smooth asymptotic curve between -1 and 1).*

*Intuition:* By looping this equation over dozens of live news headlines and taking the mean, the terminal creates a mathematical proxy for global human sentiment regarding a specific asset.
