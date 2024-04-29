Useful plots to inspect the dependence between the response variable and a given feature.  

## Challenges
These plots assume independence between the focus feature and the rest of the covariate. 

## Must know
* Partial dependence plots (PDP) show the dependence between the target response $`f(x)`$ and a set $`S`$ of input features of interest, marginalizing over the values of all other input features (the _complement_ features $`C`$).
* Similar to a PDP, an individual conditional expectation (ICE) plot shows the dependence between the target function and an input feature of interest. However, unlike a PDP, which shows the average effect of the input feature, an ICE plot visualizes the dependence of the prediction on a feature for each sample separately with one line per sample.
* In a formal notation, $`pd_{X_S}(x_S) \overset{def}{=} \mathbb{E}_{X_C}\left[ f(x_S, X_C) \right] = \int f(x_S, x_C) p(x_C) dx_C`$. The lines in the ICE plot correspond to the $`f(x_S, x_C)`$ within the integral, while their expectation is shown through the PDP.  
* For an effective visualization, the set $`S`$ may include two features at most. 

## Details
### Advanced features/remarks 
We can think of marginalizing as fixing a value for $`x_S^{fixed}`$ and then evaluating $`pd_{X_S}(x_S^{fixed}) \approx \frac{1}{n_\text{samples}} \sum_{i=1}^n f(x_S^{fixed}, x_C^{(i)})`$. However, this implies creating new samples whose coordinates are $`(x_S^{fixed}, x_C^{(i)})`$ instead of the original $`(x_S^{(i)}, x_C^{(i)})`$, which is only meaningful if S features are independent of C ones (if not, some of the created samples may not even be admissible). 

### Additional resources/info
* [sklearn implementation](https://scikit-learn.org/stable/modules/partial_dependence.html)
