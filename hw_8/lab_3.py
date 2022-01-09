# Using Bayesian Optimization to tune the hyper parameters in Scikit-Learn

# Bayesian Optimization
# pip install scikit-optimize
from lab_1 import modelDT, X_train, y_train, tree;
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

search_space = {'criterion': ['gini', 'entropy'], # default gini
    'max_depth': [3,4,5,6,7,8], # default = None means nodes are
    # expanded until all leaves are pure or until all leaves contain less
    # than min_samples_split samples
    'splitter': ['best', 'random'], # default best
    'min_samples_leaf': [1, 3, 5], # default 1
    'min_samples_split': [2, 4, 6] # default 2
}

DT_bayes_search = BayesSearchCV(
    estimator = modelDT,
    search_spaces = search_space,
    n_iter = 32,
    n_jobs = -1, # If set to -1, all CPUs are used
    random_state = 0
)

# executes bayesian optimization
DT_bayes_search.fit(X_train, y_train)

# to see number of search
DT_bayes_search.total_iterations

# to view the best parameters
DT_bayes_search.best_params_

# to display the best model
DT_bayes_search.best_estimator_

# to output the best score
DT_bayes_search.best_score_

# implement the model using the best value of the parameters
best_params = DT_bayes_search.best_params_
best_DT_model = tree.DecisionTreeClassifier(**best_params)