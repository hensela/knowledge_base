The Shapley value is a solution concept in cooperative game theory that can be extended to general ML models in order to provide an intuition of how 'important' some features are.

## Knowledge prerequisites
* [Partial Dependence](Partial Dependence and Individual Conditional Expectation plots) 

## Must know
#### Cooperative game theory
High-level recap from cooperative game theory: given a set of players, we assume that a coalition of players can cooperate and obtain a certain overall gain from that cooperation. We can formalize this better by saying there exists a function $`v`$ such that $`v(C)`$ expresses the worth of coalition $`C`$, where the worth is equal to the sum of the payoffs the members of $`C`$ can obtain through cooperation.  

Say A, B, C want to start a (small) fire, but A only has some paper, B a box of matches, C has both. If $`v`$ is binary valued (1 success, 0 failure), we would have $`v(\emptyset)=0`$, $`v(A)=0`$, $`v(B)=0`$, $`v(C)=1`$, $`v(A,B)=1`$, etc. The interesting part is that A and B clearly benefit from a cooperation and they get a better payoff compared to what they would get alone, while player C doesn't really care.  

We could ask ourselves "how much would player X gain by joining a pre-existing coalition?", but it's actually better to take the coalition's point of view: how much does player X bring to the table? If we answer this question for a fixed player and any relevant coalition, we get the Shapley value $`\varphi`$ for that player. More formally, $`\varphi_{i}(v)=\frac{1}{n !} \sum_{R}\left[v\left(P_{i}^{R} \cup\{i\}\right)-v\left(P_{i}^{R}\right)\right]`$ where the sum ranges over all $`n!`$ orders $`R`$ of the players and $`P_{i}^{R}`$ is the set of players in $`N`$ which precede $`i`$ in the order $`R`$.  

#### Shapley applied to ML 
Now... let's replace players with features, payoff with prediction, and we should have a first intuition of how cooperative game theory may be useful to allocate credit for a model’s output among its input features.  

Since in game theory a player can join or not join a game, we need a way for a feature to 'join' or 'not join' a model: usually, we say a feature has joined a model when we observe its value. When it has not, we leverage conditional expected value and integrate it out: $`E[f(X)∣X_S=x_S]`$ or $`E[f(X)∣do(X_S=x_S)]`$, where the key difference between the two equations is that the values $`x_S`$ can either be observed in the data or forcibly set leveraging assignments from a grid of values (which is computationally easier).  
When we are explaining a prediction $`f(x)`$, the Shapley value for a specific feature $`i`$ is just the difference between the expected model output and the _partial dependence_ at the feature’s value $`x_i`$.

The formulation above is model agnostic and one the fundemental properties of Shapley values is that they always sum up to the difference between the game outcome when all players are present and the game outcome when no players are present. For machine learning models this means that SHAP (SHapley Additive exPlanations) values of all the input features will always sum up to the difference between baseline (expected) model output and the current model output for the prediction being explained.  


## Details
#### Additional resources/info
* [Some theory](https://en.wikipedia.org/wiki/Shapley_value#Formal_definition)
* [Python SHAP package](https://shap.readthedocs.io/en/latest/index.html)
