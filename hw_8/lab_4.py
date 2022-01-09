# Using Bayesian Optimization to tune the hyper parameters in Scikit-Learn

# pip install scikit-optimize
from lab_1 import modelDT, X_train, y_train
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

search_space = {'criterion': ['gini', 'entropy'], # default gini
    'max_depth': [3,4,5,6,7,8], # default=None means nodes are
    # expanded until all leaves are pure or until all leaves contain less
    # than min_samples_split samples
    'splitter': ['best', 'random'], # default best
    'min_samples_leaf': [1, 3, 5], # default 1
    'min_samples_split': [2, 4, 6] # default 2
}

def on_step(optim_result):
    score = DT_bayes_search.best_score_
    print("best score: %s" % score)
    if score >= 0.98:
        print('Interrupting!')
        return True

DT_bayes_search = BayesSearchCV(modelDT, search_space, n_iter = 100, # specify how many iterations
                            scoring = "accuracy", n_jobs = -1, cv = 5)
DT_bayes_search.fit(X_train, y_train, callback = on_step) # callback = on_step will print score after each iteration

# to view the best parameters
DT_bayes_search.best_params_

# to display the best model
DT_bayes_search.best_estimator_

# to output the best score
print(DT_bayes_search.best_score_)