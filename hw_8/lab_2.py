# Using Grid Search to tune the hyper parameters in Scikit-Learn
from lab_1 import modelDT, X_train, y_train, tree;
from sklearn.model_selection import GridSearchCV

# The param_grid tells Scikit-Learn to evaluate 2 x 6 x 2 x 3 x 3 = 216 combinations of
# criterion, max_depth, splitter, min_samples_leaf and min_samples_split
# hyperparameters specified. The grid search will explore 216 combinations of
# tree.DecisionTreeClassifier’s hyperparameter values, and it will train each model 5
# times (since we are using five-fold cross-validation). In other words, all in all, there will be
# 216 x 5 = 1080 rounds of training. It may take a long time, but when it is done you can get the
# best combination of hyperparameters like this:
param_grid = {'criterion': ['gini', 'entropy'], # default gini
    'max_depth': [3,4,5,6,7,8], # default=None means nodes are expanded until all leaves are pure
    # or until all leaves contain less than min_samples_split samples
    'splitter': ['best', 'random'], # default best
    'min_samples_leaf': [1, 3, 5], # default 1
    'min_samples_split': [2, 4, 6] # default 2
}

DT_grid_search = GridSearchCV(estimator = modelDT,
                            param_grid = param_grid,
                            cv = 5,
                            scoring = "accuracy",
                            return_train_score = True,
                            verbose = 3,
                            n_jobs = -1) #If set to -1
DT_grid_search.fit(X_train, y_train)

# to get the best combination of hyperparameters
DT_grid_search.best_params_

# to get the best model
DT_grid_search.best_estimator_

# the evaluation score of the best model
DT_grid_search.best_score_

# implement the model using the best value of the parameters
best_params = DT_grid_search.best_params_
best_DT_model = tree.DecisionTreeClassifier(**best_params)