## 1 Introduction
`tpot`, Tree-Based Pipeline Optimization Tool positions itself as a "data science assistant" designed to automate construction of complex pipelines the data scientist may not have considered. `tpot` interfaces with the `scikit-learn` and `XGBoost` libraries.

[Paper Link](https://dl.acm.org/doi/pdf/10.1145/2908812.2908918)

[Project Link](http://epistasislab.github.io/tpot/)

[Github Repo](https://github.com/EpistasisLab/tpot/)

The package uses genetic search algorithms in order iterate through potential pipelines in search of the best combination of:
* Preprocessing (transformation)
* Decomposers (dimension reduction)
* Feature Selection
* Models and Algorithms

Can be applied to either regression or classification based problems

## 2 Genetic Programming
<details>
<summary>Helpful Resources</summary>


[Youtube Tutorial](https://www.youtube.com/watch?v=9zfeTw-uFCw)
- Videos 1-3 provide a strong basis for understanding genetic programming
  
[Paper Link](https://link.springer.com/chapter/10.1007/978-3-540-78293-3_22)
- Recaps the general algorithm and approach

[Helpful One Pager](http://geneticprogramming.com/tutorial/)
</details>

<details>

<summary>Basics</summary>


Genetic programming is a particular technique that aims to leverage computing power to solve problems automatically in an iterative fashion.
GP transforms populations of programs into new programs by applying specific genetic operations to pre-existing programs in hopes of finding a more ideal solution.
The process continues to create new programs throughout the system by weighing the successes of previous programs, choosing some to "breed" and applying random "mutations" or noise to a percentage of the outputs.

<summary>Key Terminology</summary>

- *Crossover* - The creation of 2 or more offspring programs by recombining randomly chosen parts from two selected parent programs
- *Mutation* - The creation of 1 new offspring program by randomly altering a randomly chosen part of one selected program
- *Crossover and Mutation are both tunable parameters in the GP system*

[Genetic Terms Diagram](https://www.sciencedirect.com/topics/medicine-and-dentistry/genetic-operator)
</details>

<details>
<summary>Basic Algorithm </summary>

1. Randomly create an initial population of programs from the available primitive possibilities
2. Execute each program and assign it a fitness score based on how well it achieves the task
3. Sample 2 outputs from step 2 (weighted by fitness score) to participate in genetic operations and combination
4. Create new "child" programs by performing genetic operations (Crossover and Mutation) on the outputs chosen above
5. Repeat steps 2-4 until some acceptable solution is found
    - Stopping condition satisfied
    - Maximium number of generations reached
6. Return the best individual program
</details>


## 3 Tpot implementation
<details>
<summary>Basic Implementation</summary>

1) Start with X random pipelines and evaluate their performance on the dataset
   - This number is a parameter which can be altered
   - A pipeline consists of a combination of preprocessing, decomposition, feature selection, and algorithm choices
2) Evaluate each pipeline using the scoring "fitness" metric
3) Take the top 20% performing pipelines and create 5 copies of each into "offspring" population
4) Perform "one point" cross-over with certain percentage of offspring population
   - Combine two elements and randomly choose elements from each to construct a new pipeline
   - Example - Take Preprocessing and Decomposition steps from pipeline A, Feature Selection and Algorithm from pipeline B
5) Mutate a certain proportion of the offspring from step 3 in one of the following ways: 
   - Point mutation: Randomly alter one step of the pipeline (ie. change model hyper-parameters or a different feature transformation method)
   - Insert a new step
   - Trim/remove an existing step
6) Repeat Steps 2-5 for Y generations
   - The more "fit" members and elements of the population have greater likelihood of staying in the population over time
7) At the end of Y generations, return the best scoring pipeline

</details>


<details>
<summary>Pros and Cons</summary>

- Pros
   - Automates large portions of the pipeline workflow
   - Tries out elements the data scientist may not think to combine in a pipeline
   - Same basic syntax as `scikit-learn` model fitting, predicting, etc.
   - Returns underlying pipeline code in Python for further configuration/adjustment
- Cons
   - No guarantee that best possible combination is returned
      - Stochastic in nature
   - Does not try to tune *all* hyperparameters for a model family
      - Example: For Random Forest Regressor, `max_depth` isn't varied
   - Long runtime
      - Out of box parameters are 100 population, 100 generations, 5 fold CV
      - Results in fitting/evaluating 50,000 models

</details>


<details>
<summary>Key Parameters</summary>

- generations
   - *int or None*
   - *default=100*
   - Number of iterations to run the pipeline for; number of times repeating the crossover-mutation-evaluation step with population members
   - If `None` - user needs to specify `max_time_mins` to set a maximum runtime
- population_size
   - *int*
   - *default=100*
   - Number of individuals (pipelines) in each round of the population
- mutation_rate
   - *default=0.9*
   - Percentage of pipelines to apply random mutations to 
- crossover_rate
   - *default=0.1*
   - Percentage of pipelines to breed together
- scoring
   - Function to optimize towards
- cv
   - *default=5*
   - Cross-Validation strategy when evaluating pipelines
- n_jobs
   - same as `scikit-learn`
   - How many processors to utilize
- subsample
   - *default=1*
   - What percentage of the training set to use during the optimization process
   - Subsample remains the same throughout the entire process
   
</details>

<details>
<summary>Algorithms Considered</summary>

- All algorithms listed below with the exception of `XGBoost Regressor` and `XGBoost Classifier` are from the `scikit-learn` library.

<details>
<Summary>Regression</Summary>

- Elastic Net
- Extra Trees Regressor
- Gradient Boosting Regressor
- Adaboost Regressor
- Decision Tree Regressor
- KNeighbors Regressor
- Lasso Lars CV Regressor
- Linear Support Vector Regressor
- Random Forest Regressor
- Ridge CV Regressor
- XGBoost Regressor
- Stochastic Gradient Descent Regressor

</details>

<details> 

<summary>Classification</summary>

- Gaussian Naive Bayes
- Bernoulli Naive Bayes
- Multinomial Naive Bayes
- Decision Tree Classifier
- Extra Trees Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- Linear Support Vector Classifier
- Logistic Regression
- XGBoost Classifier
- Stochastic Gradient Descent Classifier
- Multi-layer Perceptron Classifier

</details>

</details>
