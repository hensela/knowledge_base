Time-series $(y_t)_{t \in \mathbb{N}}$


## [Time-series components](https://otexts.com/fpp2/components.html) and [patterns](https://otexts.com/fpp2/tspatterns.html)

**Additive decomposition:**  $𝑦_𝑡 = 𝑆_𝑡 + 𝑇_𝑡 + 𝑅_𝑡$
 
Appropriate if the magnitude of the seasonal fluctuations, or the variation around the trend-cycle, does not vary with the level of the time series.

**Multiplicative decomposition:**  $𝑦_𝑡 = 𝑆_𝑡 * 𝑇_𝑡 * 𝑅_𝑡$
 
Appropriate when the variation in the seasonal pattern, or the variation around the trend-cycle, appears to be proportional to the level of the time series. 
Can be log transformed to get an additive decomposition:

$$𝑦_𝑡 = 𝑆_𝑡 * 𝑇_𝑡 * 𝑅_𝑡 \Leftrightarrow 𝑙𝑜𝑔(𝑦_𝑡) = log(𝑆_𝑡) + log(𝑇_𝑡) + log(𝑅_𝑡)$$
 
- $𝑆_𝑡$: seasonal component
- $𝑇_𝑡$: trend-cycle component
- $𝑅_𝑡$: remainder component


## [Stationarity](https://otexts.com/fpp2/stationarity.html)

<details>
<summary><b>Details</b></summary>

A stationary time series is one whose properties do not depend on the time at which the series is observed:
- constant mean ($\leftrightarrow$ no trend)
- constant standard deviation
- no seasonality

A white noise series is stationary (key property: it's not predictable):
- mean = 0
- constant standard deviation
- correlation between lags is zero

A time-series with cyclic behaviour (but no trend or seasonality) is stationary because the cycles are not of a fixed length, so before we observe the series we cannot be sure where the peaks and troughs of the cycles will be.

**Log-transformation** can help stabilize the variance of a time-series.

**Differencing** can help stabilise the mean of a time-series and eliminate/reduce trend and seasonality. 
The differenced series is the change between consecutive observations $y_t' = y_t - y_{t-1}$
- This can be applied multiple times, or there's also seasonal differencing $y_t' = y_t - y_{t-m}$ for $m \geq 1$.
- If differencing is used, it's important that the differences are interpretable.

How to test for stationarity:
- visually
- global vs. local tests
- ACF: for a stationary time-series, the ACF will drop to zero relatively quickly
- statistical hypothesis tests for stationarity to more objectively determine if differencing is required, such as the *unit root test* or *augmented Dickey-Fuller test*
</details>


## [ACF](https://otexts.com/fpp2/autocorrelation.html) and [PACF](https://otexts.com/fpp2/non-seasonal-arima.html)

<details>
<summary><b>ACF</b></summary>

Complete auto-correlation function, giving auto-correlation values of a series with its lagged values. 
Describes how much the present value of a series is related with its past values. 
A time-series can have components like trend, seasonality, cyclic and residual. 
ACF considers all these components when finding correlations.

--> Used to find the order of the moving average (MA) process
</details>

<details>
<summary><b>PACF</b></summary>

Partial auto-correlation function. 
Instead of finding correlation of present values with all lags like ACF, it finds the correlation of the residual (i.e. what remains after removing effects already explained by earlier lags) with the next lag value. 
Essentially at each time $t$ it calculates the "pure" correlations between $y_t$ and $y_{t-k}$ (for $𝑘 \geq 1$), 
removing any "indirect" effects of the type $𝑦_{𝑡−𝑘} \rightarrow 𝑦_{𝑡−𝑘+1} \rightarrow ... \rightarrow 𝑦_𝑡$ and only considering the "direct" effect $𝑦_{𝑡−𝑘} \rightarrow 𝑦_𝑡$.

The first partial autocorrelation is identical to the first autocorrelation. 
The $𝑘$-th partial autocorrelation coefficient is equal to the estimate of $\phi_𝑘$ in an $AR(𝑘)$ model.

--> Used to find the order of the auto-regressive (AR) process
</details>


## [Autoregressive model (AR)](https://otexts.com/fpp2/AR.html)

<details>
<summary><b>Details</b></summary>

An autoregressive model $AR(𝑝)$ of order $𝑝$ can be written as:

$$𝑦_𝑡 = 𝑐 + \phi_1 𝑦_{t-1} + \phi_2 𝑦_{𝑡−2} + ... + \phi_𝑝 𝑦_{𝑡−𝑝} + \epsilon_𝑡$$
 
where $\epsilon_𝑡$ is white noise. 
Essentially a linear regression with lagged values of the time-series.

Requires the time-series to be stationary.
</details>


## [Moving average model (MA)](https://otexts.com/fpp2/MA.html)

<details>
<summary><b>Details</b></summary>

Rather than using past values of the forecast variable in a regression, a moving average model uses past forecast errors in a regression-like model. 
A moving average model $MA(𝑞)$ of order $𝑞$ can be written as:

$$𝑦_𝑡 = 𝑐 + \epsilon_𝑡 + \theta_1 \epsilon_{𝑡−1} + \theta_2 \epsilon_{𝑡−2} + ... + \theta_𝑞 \epsilon_{𝑡−𝑞}$$
 
where $\epsilon_𝑡$ is white noise. 
Note that we don't actually observe the values $\epsilon_𝑡$, so it's not really a regression in the usual sense. 
Each value of $y_t$ can be thought of as a weighted moving average of the past $𝑞$ forecast errors.

It's possible to write any stationary $AR(𝑝)$ model as a $MA(\infty)$ model. 
For example for an $AR(1)$ model:

$`\begin{aligned}
𝑦_𝑡 &= \phi_1 𝑦_{𝑡−1} + \epsilon_𝑡 \\
    &= \phi_1 ( \phi_1 𝑦_{𝑡−1} + \epsilon_{𝑡−1} ) + \epsilon_t \\
    &= \phi_1^2 𝑦_{𝑡−1} + \phi_1 \epsilon_{𝑡−1} + \epsilon_𝑡 ... \\
    &= \epsilon_𝑡 + \phi_1 \epsilon_{𝑡−1} + \phi_1^2 \epsilon_{𝑡−2} + \phi_1^3 \epsilon_{𝑡−3} + ... 
\end{aligned}`$
 
The reverse holds under some constraints on the MA parameters, in which case the MA model is called **invertible**.
</details>
