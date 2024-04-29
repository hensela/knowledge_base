# Classification: K-nearest neighbors (KNN)

<details>
<summary><b>Details</b></summary>

- Calculate distances between new input all the training data points
- Sort the distances and determine the $𝑘$ nearest neighbors
- Analyze the category of those neighbors and assign the category for the new data point based on majority vote

</details>


# Classification (Regression): Support Vector Machines (SVM)

<details>
<summary><b>Resources</b></summary>

- "The Elements of Statistical Learning" (Hastie, Tibshirani, Friedman), chapter 12
- ["A Practical Guide to Support Vector Classification"](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf)
- Linear kernel: [sklearn.svm.LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
- More general implemention: [sklearn.svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)
</details>

<details>
<summary><b>Idea</b></summary>

Solving a classification problem whose data is not linearly separable. 2 approaches:

1. Allow misclassifications
2. Find a non-linear boundary by constructing a linear boundary in a higher-dimensional transformed feature space

SVM can be adapted also for regression problems (as well as multiclass classification problems)
</details>

<details>
<summary><b>Option 1: Allow misclassifications</b></summary>

Set a "margin" that should ideally be achieved in the separation of the 2 classes. 
Penalize every training point that's within this margin (both on the correct side or on the wrong side of the separating line). 
This leads to a new constraint for the optimization objective.
The sensitivity (i.e. the size of this margin) can be specified. 
An infinite sensitivity corresponds to the linearly separable case, where a low sensitivity also takes into consideration data further away from the decision boundary. 
In this way, SVM can also make a better bias-variance tradeoff.
</details>

<details>
<summary><b>Option 2: Kernel methods</b></summary>

The idea is to lift the feature space up to a higher dimensional space by applying transformation functions. 
If $`\left\{ (x_i, y_i) \right\}_{i=1}^N`$ are the training data points and we have a transformation function $h$, 
then one can try to find a linear decision boundary for the higher-dimensional data $`\left\{ (h(x_i), y_i) \right\}_{i=1}^N`$.

To solve this for more complex transformations (to possibly infinite dimensionsal spaces), SVM relies on the kernel trick: 
Due to the mathematical formulation of the problem, one finds that we don't have to perform the actual feature transformations themselves (nor even specify the transformation function $h$), 
but rather the optimization problem to solve requires only knowledge of the kernel function $K(x, x') = \langle h(x), h(x') \rangle$ that computes the inner products in the transformed space.

Three popular choices for $K$ in the SVM literature are:

- $d \text{th}$-Degree polynomial:  $K(x, x') = (1+ \langle x, x' \rangle)^d$
- Radial basis:  $K(x, x') = exp(−\gamma ||x−x'||^2)$
- Neural network:  $K(x, x') = tanh( \kappa_1 \langle x, x' \rangle + \kappa_2)$
</details>


# Classification: Logistic Regression

<details>
<summary><b>Details</b></summary>

Training observations $`\left\{ (x(i),y(i)) \right\}_{i=1}^N`$, each of which having $m$ features $x^i = \left( x_1^i,..., x_m^i \right) \in \mathbb{R}^m$.
We fit a linear regression model:

$$z_i = \theta_0 + \theta_1 x_1^i +...+ \theta_𝑚 x_m^i$$
 
Our prediction will be (sigmoid function):

$$h_{\theta} \left( x^i \right) = \frac{1}{1 + e^{−z^i}}$$

The cost function to use is the log-loss / binary cross-entropy:

$$J(\theta) = −\frac{1}{N} \sum_{i=1}^{N} { [ y_i log( h_{\theta} (x^i) + (1−y_i) log(1 − h_{\theta}(x^i) ) ]$$
 
We cannot use mean squared error as a loss function, as it's non-convex in this case and has many local minima.
</details>

