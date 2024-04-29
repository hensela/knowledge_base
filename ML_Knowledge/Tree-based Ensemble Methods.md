# Brief overview of tree-based ensemble methods

**Assumed knowledge:**
- How decision trees work for both classification and regression (see also [here](https://gitlab.criteois.com/ax-analytics/gads/knowledge/-/blob/master/ml_models/ClassificationTrees.ipynb))

**Introductory resources:**
- [The Ultimate Guide to AdaBoost, random forests and XGBoost](https://towardsdatascience.com/the-ultimate-guide-to-adaboost-random-forests-and-xgboost-7f9327061c4f)
- [Gradient Boosting Trees for Classification: A Beginner’s Guide](https://medium.com/swlh/gradient-boosting-trees-for-classification-a-beginners-guide-596b594a14ea)
- [Tree-Based Models: How They Work (In Plain English!)](https://blog.dataiku.com/tree-based-models-how-they-work-in-plain-english)


## 0 Introduction
The idea behind _ensemble methods_ is to combine a set of _weak_ lerners to form a _strong_ learner. 
A _weak_ lerner refers to an algorithm that only predicts slightly better than randomly. 
For tree-based ensemble algorithms a single decision tree would be the weak learner and a tree-based ensemble algorithm combines multiple of these.

For tree-based algorithms, there are two basic approaches to combine weak lerners:
- **Bagging:** Bootstrap Aggregation or Bagging is a ML algorithm in which a number of independent predictors are built by taking samples with replacement of the training set. The individual outcomes are then combined by average (Regression) or majority voting (Classification) to derive the final prediction.
- **Boosting:** In Boosting, the weak predictions are sequential wherein each subsequent weak predictor learns from the errors of the previous predictors. The final prediction is based on a weigthed sum of the individual predictions, where the weak predictors may have different weights.

Some general advantages of tree-based ensemble methods:
- can handle mixed data types
- can handle differently scaled input data
- generally robust against overfitting
- generally handle noisy data and outliers well
- can handle multi-colinearity of features (i.e. features don't necessarily have to be removed to decrease interactions between them)
- reasonably well understandable and interpretable

## 1 Bootstrap-and-aggregation ("Bagging") algorithms

### 1.1 Random Forest
<details>
  <summary>Resources and Implementations</summary>

  Resources:
  - [Original paper](https://link.springer.com/content/pdf/10.1023/A:1010933404324.pdf) (Breimann, 2001)
  - [Youtube video series](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ) explaining Random Forest

  Implementations:
  - [sklearn.ensemble.RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
  - [sklearn.ensemble.RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

</details>

<details>
  <summary>Algorithm</summary>
  
  **Input:** 
  - training dataset $`\mathcal{D}`$
  - integer $`M`$ specifying number of trees in the forest
  - integer $`s`$ specifying the size of the bootstrap sample of training datapoints used for building each tree
  - integer $`f \leq |\mathcal{F}|`$ specifying the number of input features to consider at each node for each tree, where $`\mathcal{F}`$ is the set of all input features
  
  **Iteration:** build up $`M`$ separate trees $`(m = 1,..., M)`$ as follows:
  - Select a _bootstrap sample_ $`\mathcal{S}`$ of size $`s`$ of the training data (i.e. draw $`s`$ random samples with replacement from $`\mathcal{D}`$)
  - Create the tree $`m`$ by recursively repeating the following steps for each internal node of the tree (until the tree’s prediction does not further improve):
      1. Randomly choose $`f`$ features from the set of all available features $`\mathcal{F}`$
      2. Select the best feature among the $`f`$ chosen ones (i.e. the one with the most information gain)
      3. Use this feature to split the current node of the tree on
  
  
  **Output:**
  Majority voting of all $`M`$ trees decides on the final prediction results
</details>

<details>
  <summary>Advantages and Disadvantages</summary>
  
  - (+) relatively robust against noise and outliers
  - (+) can do implicit feature selection
  - (+) it's fast
  - (-) many relevant hyperparameters to tune
  - (-) introduces randomness, which may not be suitable for all datasets
</details>

## 2 Boosting algorithms

### 2.1 AdaBoost
_AdaBoost stands for "Adaptive Boosting"_

<details>
  <summary>Resources and Implementations</summary>
  
  Resources:
  - [Original paper](https://pdfs.semanticscholar.org/5fb5/f7b545a5320f2a50b30af599a9d9a92a8216.pdf) (Freund and Schapire, 1996)
  - Nice [Youtube video](https://www.youtube.com/watch?v=LsK-xG1cLYA) explaining AdaBoost
  
  Implementations:
  - [sklearn.ensemble.AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
  - [sklearn.ensemble.AdaBoostRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)
</details>

<details>
  <summary>Basic ideas behind AdaBoost (for binary Classification)</summary>
  
  - Assign _weights_ to training observations
  - Train weak learner on weighted training observations
  - Assign _weight_ to the trained weak learner and re-weight the training observations based on the errors the weak learner makes
  - _Normalize_ weights to sum up to 1 and train the next weak learner
  - The final prediction is based on a weighted sum of the predictions of the weak learners
  
  The algorithm can be adapted to tackle regression problems as well.
</details>

<details>
  <summary>Algorithm (for binary classification)</summary>
  
  **Input:** 
  - sequence of $`N`$ labeled examples $`\{(x_1, y_1),..., (x_N, y_N)\}`$
  - distribution $`D`$ over the $`N`$ examples
  - weak learning algorithm of choice `WeakLearn` (for example "finding the best decision tree of depth 1 based on weighted Gini score")
  - integer $`M`$ specifying number of iterations
  
  **Initialization:** 
  - weight vector: $`w_i^1 = D(i)`$ for $`i = 1,..., N`$ (common initialization is a uniform distribution $`D`$, s.t. each example is initialized with the weight $`1/N`$)
  
  **Iteration:** for $`m = 1,...,M`$
  1. Normalize the weight vector $`\mathbf{w}^m`$ by setting: $`\mathbf{p}^m = \mathbf{w}^m \big/ \sum_{i}{w_i^m}`$
  2. Call `WeakLearn`, providing it with the distribution $`\mathbf{p}^m`$; get back a hypothesis $`h_m: X \rightarrow [0, 1]`$
  3. Calculate the error of $`h_m`$ as: $`\epsilon_m = \sum_{i}{p_i^m \left| h_m(x_i) - y_i \right|}`$
  4. Calculate the weight of weak learner $`m`$ as (mind the difference between "w" and "omega"): 
     $$
     \omega_m = \frac{1}{2} log \left( \frac{1 - \epsilon_m}{\epsilon_m} \right)
     $$
  5. Set the new weights vector to be 
     $$
     p_i^{m+1} = p_i^m e^{\pm \omega_m}
     $$

     where the sign in the exponent is positive for incorrectly classified samples and negative for correctly classified samples
  
  
  **Output:**
  The hypothesis $`h_f(x)`$ with
  - $`h_f(x) = 1`$ if $`\sum_m{\omega_m} h_m(x) \geq \frac{1}{2} \sum_m{\omega_m}`$ (equivalently: the prediction is positive if the sum of weights of weak classifiers that predict $`h_m(x)=1`$ is greater or equal than the sum of weights of weak classifiers that predict $`h_m(x)=0`$)
  - $`h_f(x) = 0`$ otherwise
</details>
  
<details>
  <summary>Advantages and disadvantages</summary>
  
  - (+) relatively robust to overfitting in low noise datasets
  - (+) only few hyperparameters that need to be tuned to improve model performance
  - (-) for noisy data the performance of AdaBoost is debated
  - (-) compared to random forests and XGBoost, AdaBoost performs worse when irrelevant features are included
  - (-) not optimized for speed
</details>


### 2.2 Gradient Boost (GBM)

<details>
  <summary>Resources and Implementations</summary>
  
  Resources:
  - [Original paper](http://statweb.stanford.edu/~jhf/ftp/stobst.pdf) (Friedman, 1999)
  - [Another paper](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf) (Friedman, 2001)
  - [Youtube video series](https://www.youtube.com/watch?v=3CC4N4z3GJc&t=3s) explaining Gradient Boost
  
  Implementations:
  - [sklearn.ensemble.GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
  - [sklearn.ensemble.GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
</details>

<details>
  <summary>Algorithm</summary>
  
  **Input:** 
  - sequence of $`N`$ labeled examples $`\{(x_1, y_1),..., (x_N, y_N)\}`$
  - differentiable Loss function $`L(y_i, F(x))`$
  - integer $`M`$ specifying number of iterations
  
  **Initialization:** Initialize model with a constant value: $`F_0(x) = \underset{\gamma}{argmin} \sum_{i=1}^{N}{L(y_i, \gamma)}`$.
  Note that if $`L`$ is the mean squared error (MSE) function (with a factor $`\frac{1}{2}`$), then $`\gamma = \frac{1}{N} \sum_{i=1}^{N}{y_i}`$ is just the average value.
  
  **Iteration:** for $`m = 1,...,M`$
  1. Compute the _pseudo residuals_ for $`i=1,..., N`$:
  $$
  r_{i,m} = - \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \Bigg|_{F(x)=F_{m-1}(x)}
  $$
  Note that $`L`$ is the MSE function, then the $`r_{i,m} = y_i -  F_{m-1}(x_i)`$ are just the residuals.
  
  2. Fit a regression tree to predict the $`\{ r_{i,m} \}_{i=1}^{N}`$ and compute terminal regions (leaves) $`R_{j,m}`$ for $`j=1,..., J_m`$ (i.e. the number of leaves in the weak classifier fitted to the $`\{ r_{i,m} \}_{i=1}^{N}`$ is $`J_m`$)
  
  3. For $`j=1,..., J_m`$ compute 
     $$
     \gamma_{j,m} = \underset{\gamma}{argmin} \sum_{x_i \in R_{j,m}}{L(y_i, F_{m-1}(x_i) + \gamma)}
     $$
     Again, if $`L`$ is MSE, then $`\gamma_{j,m}`$ is just the average of the residuals that ended up in leaf $`R_{j,m}`$.
  
  4. Update 
     $$
     F_m(x) = F_{m-1}(x) + \nu \sum_{j=1}^{J_m}{\gamma_{j,m} \mathbf{I}(x \in R_{j,m})}
     $$

     where $`\nu \in (0, 1)`$ is called _learning rate_ and $`\mathbf{I}`$ is just the characteristic function.
  
  **Output:** $`F_M(x)`$
</details>

<details>
  <summary>Advantages and Disadvantages</summary>
  
  - (+) relatively robust to overfitting
  - (+) general formulation that works with any differentiable loss function (AdaBoost is a special case of GBM using a specific loss function)
  - (-) more prone to be thrown off by irrelevant features than random forests and XGBoost
  - (-) sensitive to outliers
  - (-) more hyperparameters to tune
</details>


### 2.3 XGBoost
_XGBoost stands for "Extreme Gradient Boosting"_

<details>
  <summary>Resources and Implementations</summary>
  
  Resources:
  - [Original paper](https://arxiv.org/pdf/1603.02754.pdf) (Chen and Gestrin, 2016)
  - [Youtube video series](https://www.youtube.com/watch?v=OtD8wVaFm6E) explaining XGBoost
  
  Implementations:
  - `xgboost` [Python package](https://xgboost.readthedocs.io/en/latest/python/index.html)
</details>

<details>
  <summary>Algorithm (main ideas)</summary>
  
  In essence, XGBoost can be seen as enhancing Gradient Boost by adding regularization terms.
  Similar to Gradient Boost, XGBoost also uses a _learning rate_ or _shrinkage parameter_ $`\nu`$ to further prevent overfitting.
  In addition to that, like random forests, XGBoost also allows subsampling of features and bootstrapping of training observations, making it less prone to be thrown off by irrelevant features.
  
  The optimization objective for Gradient Boost seen above can be formulated as follows: In the $`m`$-th iteration the $`F_m`$ is found that minimizes the following objective:
  $$
  \mathcal{L}_m = \sum_{i=1}^{N}{L(y_i, F_{m-1}(x_i) + F_m(x_i))}
  $$
  
  XGBoost adds regularization terms, leading to the following optimization objective in the $`m`$-th iteration:
  $$
  \mathcal{L}_m = \sum_{i=1}^{N}{L(y_i, F_{m-1}(x_i) + F_m(x_i))} + \gamma T_m + \lambda ||\mathbf{w}_m||^2
  $$
  where:
  - $`\gamma \geq 0`$ is a penalty parameter and $`T_m`$ is the number of leaves in the $`m`$-th tree. Hence, the term $`\gamma T_m`$ is meant to encourage pruning of the tree (i.e. penalize trees with too many leaves)
  - $`\lambda\geq 0`$ is a regularization parameter
  - $`\mathbf{w}_m \in \mathbb{R}^{T_m}`$ is a vector of _leaf weights_ for the $`m`$-th tree (that also have to be found as part of the optimization objective)
  
  As this optimization objective is harder to solve for general loss functions, what's done in the original paper is to use a second order Taylor approximation around $`F_m(x_i)=0`$ for the term $`L(y_i, F_{m-1}(x_i) + F_m(x_i))`$ to pull $`F_m(x_i)`$ out of the loss function (for $`i=1,..., N`$).
</details>

<details>
  <summary>Advantages and Disadvantages</summary>
  
  - (+) XGBoost can be seen as combining the best elements of Gradient Boost and random forests
  - (+) regularization prevents overfitting
  - (+) subsampling from features makes it less prone to be thrown off by irrelevant features
  - (+) it's fast
  - (-) it's harder to understand and tune (more hyperparameters)
</details>


### 2.4 CatBoost
_CatBoost stands for "unbiased boosting with categorical features"_

<details>
  <summary>Resources and Implementations</summary>
  
  - [CatBoost documentation](https://catboost.ai)
  - [Original paper](https://arxiv.org/pdf/1706.09516.pdf)
  - Very short [Youtube video](https://www.youtube.com/watch?v=jLU6kNRiZ5o) summarizing the key findings of the CatBoost paper
  - [Blog post](https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db) on CatBoost vs. Light GBM vs. XGBoost
  - [Blog post](https://towardsdatascience.com/why-you-should-learn-catboost-now-390fb3895f76) focused on how to work with CatBoost and its strengths
</details>

<details>
  <summary>Algorithm (main ideas)</summary>
  
  **Unbiased boosting**
  
  The authors of CatBoost realized a problem that pertains to all previous gradient boosting methods, namely a form of _target leakage:_
  
  Consider step 1 of the Gradient Boost algorithm outlined above.
  The next model $`F_m`$ (for $`m \geq 1`$), relies on the _pseudo residuals_ or _gradients_ $`\{ r_{i,m} \}_{i=1}^N`$.
  However, the gradients used at each step $`m \geq 1`$ are estimated using the **target values** of the same data points the current model $`F_{m-1}`$ was built on.
  They prove that this can introduce a bias in the prediction (they make a specific example with Bernoulli random variables, where they show that the bias term is proportional to $`1/N`$).
  
  CatBoost solves this problem using a version of _ordered boosting:_
  In this procedure, at every step $`m = 1,...,M`$ there are $`N`$ _supporting models_ $`M_1,...,M_N`$ maintained.
  These supporting models are such that the model $`M_i`$ is learned using only the first $`i`$ examples in the training set. 
  At each step, in order to obtain the gradient for $`j`$-th sample, the model $`M_{j-1}`$ is used (which itself has been build **without** relying on the target value the $`j`$-th training example).
  
  To further reduce variance, CatBoost doesn't rely on just one given ordering of the training samples, but each new tree is built using a permutation $`\sigma`$ of the training samples, randomly sampled from a set of $`s`$ permutations $`\{ \sigma_r \}_{r=1}^s`$ (the permutations $`\sigma_r`$ are random, but once selected they are fixed).
  Thus, we have a set of $`\{ M_{r,i} \, | \, 1 \leq r \leq s, \, 1 \leq i \leq N \}`$ supporting that have to be maintained.
  
  
  **Using categorical features**
  
  One of CatBoost's strengths is that it can work directly with categorical features, without the need of first having to manually encode them (e.g. using one-hot-encoding).
  The starting point of CatBoost's approach is an encoding technique for categorical features called _target encoding_ or _likelihood encoding:_
  
  - If the target variable to be predicted is categorical itself (i.e. for a classification problem), the idea of _target encoding_ is to replace each value of a categorical feature with a number calculated from the distribution of the target labels for that particular value of the categorical feature. For example, for a categorical feature "colour" that takes values red, blue and green, replace red with some statistical aggregate (mean, median, variance etc.) of all the target labels where the feature value is red in training data. Similarly for blue and green.
  - This procedure is adapted for regression problems by _quantizing_ the target values and putting them into a finite number of discrete "buckets" (labelled with integer values). Then the same procedure for _target encoding_ as described above can be applied.
  - This method of encoding categorical features is particularly useful for _high-cardinality_ categorical features taking a high number of discrete values (consider for example a "user id" feature). In this case a one-hot-encoding strategy can lead to an unfeasible amount of new features, whereas the target encoding strategy still only creates a single feature. Moreover, the approach also works if the categorical value of a test example has never been observed before in a training example (e.g. a new user id).
  
  However, the authors of CatBoost realized a similar _target leakage_ issue with the target encoding strategy as they did for the boosting procedure.
  Namely, the value of the target encoding of a training example is calculated using this training examples target label.
  Similar to above, their approach to solve this problem relies on the _ordering principle:_
  For each training example they only use previous training examples to calculate its target encoding.
  As above, to further reduce variance they don't rely on just a single ordering of the training examples, but use a set of permutations.
  
  To learn more about this encoding and how it's implemented in CatBoost:
  - [CatBoost: Transforming categorical features to numerical features](https://catboost.ai/docs/concepts/algorithm-main-stages_cat-to-numberic.html)
  - [Blog post](https://towardsdatascience.com/getting-deeper-into-categorical-encodings-for-machine-learning-2312acd347c8) explaining encodings for categorical features, in particular target encoding
</details>

<details>
  <summary>Advantages and disadvantages</summary>
  
  - (+) CatBoost has been shown to outperform other boosting algorithms (such as XGBoost, LightGBM) on a variety of datasets
  - (+) the ordering approach approach of CatBoost has been found to be especially useful for small datasets
  - (+) handles categorical features well internally, without the need for manual feature transformation beforehand
  - (+) requires only little hyperparameter tuning
  - (+) it's fast
  - (+) the CatBoost package seems to have good capabilities to visualize both feature importance as well as the importance of individual training examples on the predictions (hopefully more to come on that, once I get to play around with it myself)
  - (-) it's more complex
</details>


## Further Topics

### Forcing monotone dependence on a feature
In tree-based models it's possible to force the prediction to have a monotone dependence on an input feature (with index $`i`$), so that for any two observations $`x^{(k)}, x^{(l)}`$ with $`x^{(k)}_i \geq x^{(l)}_i`$ and $`x^{(k)}_j = x^{(l)}_j, \forall j \neq i`$, the predictions fulfill $`\hat{y}^{(k)} \geq \hat{y}^{(l)}`$ (or similarly for $`\leq`$ instead of $`\geq`$).
The post [here](https://towardsdatascience.com/how-does-the-popular-xgboost-and-lightgbm-algorithms-enforce-monotonic-constraint-cf8fce797acb) explains how this is implemented in some algorithms.
To summarize how it works:
- For any tree that is built, for any node which is split on feature $`i`$, any split value resulting in a weight $`w_L`$ of the left child to be higher than the weight $`w_R`$ of the right child is abandoned (i.e. its loss is set to infinity). Here "weight" means the value that would be predicted at a node, if it would be considered a leaf. This ensures monotony at node-level.
- To ensure monotony at tree-level, one has to take care of the case of feature $`i`$ being used multiple times throughout the tree. Let's consider feature $`i`$ being used a second time to split the left child (L) of a parent node (P) that was split using feature $`i`$. Then, first the weight $`w_{LL}`$ of the left child LL of L and the weight $`w_{LR}`$ of the right child LR of L are required to fulfill $`w_{LL} \leq w_{LR}`$ by the first point above. In addition, they are both forced to be bounded by the mean of the weights $`m := mean(w_L, w_R)`$ of the right and left node of P, that is they must fulfill $`w_{LL}, w_{LR} \leq m`$ .
- The above conditions ensure monotony at tree-level. Since the ensemble prediction is a sum of tree leaf node weights, monotony is ensured also at algorithm-level.

It's worth noting that the above conditions are somewhat restrictive. 
While they may improve performance in cases where monotony is well-motivated, the may be detrimental in other cases.
