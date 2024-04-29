Time-series $(y_t)_{t \in \mathbb{N})$.


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