# 🧮 The Mathematics of the Quant Terminal

This document outlines the statistical theories, mathematical formulas, and quantitative intuition powering the core analytical modules of the Multi-Asset Quantitative Strategy Terminal.

## Tab 1: Classic Statistical Forecasting (ARIMA & ETS)
This tab relies on traditional time-series mathematics, which assumes that the future is a mathematical function of the past.

### 1. ARIMA (AutoRegressive Integrated Moving Average)
ARIMA predicts future prices by looking at its own past prices and past errors. It is defined by three parameters: $(p, d, q)$.
* **$AR(p)$ - AutoRegressive:** The model looks at $p$ previous days. If $p=5$, the model assumes today's price is a linear combination of the last 5 days.
* **$I(d)$ - Integrated:** Financial data is usually "non-stationary" (it trends infinitely upward or downward). We difference the data $d$ times (subtracting yesterday's price from today's) to make the math stable around a baseline of zero.
* **$MA(q)$ - Moving Average:** The model looks at the error (the difference between its prediction and reality) of the last $q$ days to smooth out sudden "shocks" in the market.

**The Equation:**
$$Y_t = c + \phi_1 Y_{t-1} + \dots + \phi_p Y_{t-p} + \theta_1 \epsilon_{t-1} + \dots + \theta_q \epsilon_{t-q} + \epsilon_t$$
*(Where $Y_t$ is the differenced price, $\phi$ represents past price weights, and $\theta$ represents past error weights).*

### 2. Exponential Smoothing (ETS)
Unlike a simple moving average that treats all past days equally, ETS gives exponentially decreasing weight to older observations. Yesterday is mathematically vastly more important than a day from three years ago.

---

## Tab 2: Algorithmic & Machine Learning
This tab steps away from classic linear statistics and uses modern, non-linear algorithmic approaches.

### 1. Facebook Prophet
Prophet does not look at the market as a sequence of steps; it looks at it as a curve made of overlapping waves. It uses a decomposable additive time-series model.

**The Equation:**
$$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$
* **$g(t)$:** The core trend (is the asset generally going up or down?).
* **$s(t)$:** Seasonality (e.g., human behavior like "Crypto drops on Sundays").
* **$h(t)$:** Holiday effects (anomalous days).

*Intuition:* Prophet is highly resilient to missing data (like weekends) and sudden trend shifts, making it exceptional for 24/7 assets like Bitcoin.

### 2. Random Forest Regressor
Instead of using time-series math, this uses an ensemble of Decision Trees. We extract "calendar features" (Day=15, Month=10, Year=2023) and ask the algorithm: *"Historically, when it is the middle of October, what does this asset usually do?"* It builds hundreds of hypothetical decision trees and averages their guesses together to prevent overfitting.

---

## Tab 3: Multivariate Macro SARIMAX
This is the holy grail of macroeconomic forecasting. SARIMAX takes the standard ARIMA model and introduces an **Exogenous Variable ($X$)**. 

Instead of just looking at Apple's past prices to predict Apple's future, SARIMAX mathematically factors in a completely separate, independent driver (like the US Dollar or Interest Rates). 

**The Equation:**
$$Y_t = \text{ARIMA}(p,d,q) + \beta X_t$$
*Intuition:* The $\beta X_t$ term represents the outside macroeconomic force. If we use Crude Oil to predict an Airline stock, the model mathematically calculates how heavily a spike in oil prices will drag down the airline's future valuation.

---

## Tab 4: Macro Correlation Scanner
Before running SARIMAX, you must know *which* exogenous variable to use. This tab calculates the **Pearson Correlation Coefficient** for a matrix of global assets.

**The Equation:**
$$\rho_{X,Y} = \frac{\text{cov}(X,Y)}{\sigma_X \sigma_Y}$$
*Intuition:* The math measures the linear relationship between two assets. It outputs a score between **-1.0** and **+1.0**.
* If Asset A and Asset B have a correlation of **+0.90**, they move perfectly together.
* If they have a correlation of **-0.85**, they move in perfect opposite directions (e.g., when the US Dollar spikes, Gold crashes). The scanner visualizes this as a heatmap to allow the user to select the darkest-colored variable for their SARIMAX model.

---

## Tab 5: Modern Portfolio Theory (MPT) & The Optimizer

This is the most mathematically rigorous section of the terminal, utilizing **Quadratic Programming**. Pioneered by Harry Markowitz, MPT proves that by combining risky assets that are not perfectly correlated, you can mathematically reduce the overall risk of the portfolio without sacrificing returns.

### The Objective: Maximize the Sharpe Ratio
The algorithm's sole purpose is to find the exact allocation weights ($w$) of capital to maximize the Sharpe Ratio (return per unit of risk).

**The Sharpe Ratio Equation:**
$$S = \frac{R_p - R_f}{\sigma_p}$$
*(Where $R_p$ is expected portfolio return, $R_f$ is the risk-free rate, and $\sigma_p$ is portfolio volatility).*

### The Mathematical Components
To calculate the Sharpe Ratio, the algorithm must calculate two massive matrices for the portfolio:

**1. Expected Portfolio Return ($R_p$):** The weighted average of individual asset returns.
$$R_p = \sum_{i=1}^{n} w_i \mu_i$$

**2. Portfolio Variance / Risk ($\sigma_p^2$):**
This is where the quadratic math happens. Risk is not just the average volatility of the assets; it must account for how the assets interact with each other (Covariance). 
$$\sigma_p^2 = w^T \Sigma w$$
*(Where $w$ is the vector of asset weights, $w^T$ is its transpose, and $\Sigma$ is the covariance matrix of the assets).*

### The Constraints (SLSQP Algorithm)
We use the SciPy library to run Sequential Least Squares Programming. We ask the computer to test thousands of combinations to maximize the Sharpe Ratio, but we force it to obey two rules:
1.  $\sum w_i = 1$ (You must invest exactly 100% of your capital).
2.  $w_i \ge 0$ (No short-selling; weights cannot be negative).

---

## Tab 6: Vectorized Algorithmic Backtesting
This module tests a systematic Quantitative Trading strategy (Moving Average Crossover) without using any loops, relying entirely on matrix vectorization for speed.

**The Core Logic:**
$$SMA_t = \frac{1}{n} \sum_{i=0}^{n-1} P_{t-i}$$
If $SMA_{fast} > SMA_{slow}$, the Signal = 1 (Buy). 

**The Lookahead Bias Prevention:**
In quantitative programming, you cannot trade today based on today's closing price, because you do not know the closing price until the market closes. Therefore, the code mathematically forces a 1-day delay:
$$Position_t = Signal_{t-1}$$
By shifting the signal vector forward, the backtester simulates real-world execution, ensuring the algorithm's reported ROI is historically accurate and not synthetically inflated.
