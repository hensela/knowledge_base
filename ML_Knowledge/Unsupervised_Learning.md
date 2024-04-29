# Principal Component Analysis (PCA)
Basically, PCA relies on the singular value decomposition of the matrix of data points, which is of the form  $𝑋=𝑈𝐷𝑉^𝑇$, where both U and V are orthogonal matrices and the first m columns of V are the first m principal components.

## Idea of the algorithm:

1. Center the data points by subtracting the mean
2. Do a linear regression using sum of squared errors (SSE) as loss function
3. Note that SSE can be seen in two equivalent ways: 
One way is to minimize the sum of squared distances of the data points to the fitted line, let's call this  $𝑎_𝑖$ for a data point  $𝑥_𝑖$
(that's already been shifted by the mean of data points). 
Let  $𝑐_𝑖=||𝑥_𝑖||$ be the distance of  $𝑥_𝑖$ to the origin. 
Then, by Pythagoras, the value $𝑏_𝑖$ that solves the equation  $𝑐_𝑖^2=𝑎_𝑖^2+𝑏_𝑖^2$ is the distance of the projection of  $𝑥_𝑖$ onto the fitted line to the origin. 
One observes that minimizing the sum of squared errors  $\sum_i{𝑎_𝑖^2}$ corresponds to maximizing the sum $\sum_i{b_𝑖^2}$. 
The second sum is interpreted as the variance of the data along the fitted line. 
Hence the fitted line, corresponds to the direction along which the variance of the data is the largest, which is called the first principal component.
4. To find the next principal component, the restriction of it having to be orthogonal to all the previous ones is added.