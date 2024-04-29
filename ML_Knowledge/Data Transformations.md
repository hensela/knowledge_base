## Transforming the target variable

Resources:
- https://florianwilhelm.info/2020/05/honey_i_shrunk_the_target_variable/
- https://davegiles.blogspot.com/2014/12/s.html?m=1

**Avoid introducing bias:**

Care must be taken when applying a non-linear transformation $h$ to the target variable $y$ and fitting the model to predict $𝑧 := h(𝑦)$.
Depending on the distribution of the target variable and the error metric to be minimized in the original un-transformed space, 
the optimal prediction in the original space is not necessarily given by $h^{−1}(\hat(z))$ (where $\hat(z)$ is the prediction in the transformed space).

To understand this, assumptions are made on the distribution of the error $z - \hat(z)$ the model makes in the transformed space 
(usually that it follows a normal distribution with mean  0). 
Along with that also assumptions are made on the distribution of $y$ (if a log-transformation is to be applied, usually that it's log-normal distributed) 
and thus also the distribution of $\hat(y)$ (since we already make assumptions on the distribution of the errors).

To see where the error metric comes in, one actually considers the distribution of $y$ conditioned on an input vecor $x$.
With this, one can work out that if predictions are made in the original space, for example in the case of MSE, the prediction $\hat(y)_i$
(where $i$ is the index of an observation with input vector $x_i$) is the expected value of $y | x_i$, 
whereas in the case of MAE, the prediction $\hat{y}_i$ is the median of $y | x_i$.

Knowing the transformation and the metric to be minimized in the original un-transformed space, one can then work out how theoretically the predictions $\hat(z)_i$
have to be shifted (by an additive factor, in case of log-transformation and log-normal assumptions) in the transformed space before back-transforming to the original space, 
in order to predict the optimal value as defined by the error metric.

Finding the theoretical value requires assumptions to be made on the distributions, along with calculations depending on the transformation function and the error metric. 
While there are non-parametric approaches to this re-transformation problem (e.g. Duan's smearing factor), it's probably easier to find the value of the shift to be applied before the back-transformation via cross-validation, rather than working it out theoretically.

**Connection to the loss function:**

Let's consider the case of log-transforming the target variable $y$ and fitting a model $h_{\theta}$ with parameters $\theta$ to predict $𝑙𝑜𝑔(𝑦)$ using features $x$.

If we use the MAE (mean absolute error) loss function in the log-space, the optimization problem being solved when fitting the model translates to finding the optimum w.r.t. MLAR ("mean log accuracy ratio") in the untransformed space:

$`\begin{aligned} 
\underset{\theta}{\mathrm{argmin}} MAE(𝑙𝑜𝑔(𝑦), h_{\theta}(𝑥)) &= \underset{\theta}{\mathrm{argmin}} \frac{1}{n} \sum_i{ \left| log(y_i) - log(e^{h_{\theta}(x_i) }) \right| }\\
    &= \underset{\theta}{\mathrm{argmin}} \frac{1}{n} \sum_i{ \left| log \left( \frac{y_i}{e^{h_{\theta}(x_i)}} \right) \right| } \\
    &= MLAR \left( y,  e^{h_{\theta}(x_i)} \right) \\
    &\approx MAPE \left( y,  e^{h_{\theta}(x_i)} \right) \text{ } \text{ (same Taylor series argument as below)}
\end{aligned}`$ 


On the other hand, using the (R)MSE loss function in the log-space (as the square root is a monotone function, the argmin of RMSE and MSE are the same), 
we rediscover an approximation to the argmin of the MSPE (mean squared percentage error) in the untransformed space:

$`\begin{aligned} 
\underset{\theta}{\mathrm{argmin}} (R)MSE(𝑙𝑜𝑔(𝑦), h_{\theta}(𝑥)) &= \underset{\theta}{\mathrm{argmin}} \frac{1}{n} \sum_i{ \left( log(y_i) - log(e^{h_{\theta}(x_i) }) \right)^2 }\\
    &= \underset{\theta}{\mathrm{argmin}} \frac{1}{n} \sum_i{ \left( log \left( \frac{y_i}{e^{h_{\theta}(x_i)}} \right) \right)^2 } \\
    &\approx \underset{\theta}{\mathrm{argmin}} \frac{1}{n} \sum_i{ \left( 1 - \frac{y_i}{e^{h_{\theta}(x_i)}} \right)^2 } \text{ } \text{ (Taylor series expansion)}
    &\approx \underset{\theta}{\mathrm{argmin}} MSPE \left( y,  e^{h_{\theta}(x_i)} \right)
\end{aligned}`$ 

