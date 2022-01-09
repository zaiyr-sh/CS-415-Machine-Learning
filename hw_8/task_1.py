import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

data = pd.read_csv('hw_8/data.csv')

data.pop('id')
y = data.pop('diagnosis').values
X = data.values

X = np.nan_to_num(X.astype(np.float32))

# split the data using Scikit-Learn's train_test_split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.75, test_size = 0.25, random_state = 1)

svr = SVC()
svr.fit(x_train, y_train)

# y_pred = svr.predict(x_test)

score = svr.score(x_test, y_test)
print(round(score, 2))

from sklearn.metrics import accuracy_score
y_pred = svr.predict(x_test)
print("Accuracy: ", round(accuracy_score(y_test, y_pred), 2))

param_grid = {'C': [0.1, 1, 10, 100, 1000],
    'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5],
    'kernel': ['rbf', 'linear']
}

DT_grid_search = GridSearchCV(estimator = svr,
                            param_grid = param_grid,
                            cv = 5,
                            scoring = "accuracy",
                            return_train_score = True,
                            verbose = 3,
                            n_jobs = -1) #If set to -1
DT_grid_search.fit(x_train, y_train)

# to get the best combination of hyperparameters
print('The best combination of hyperparameters:', DT_grid_search.best_params_)
# The best combination of hyperparameters: {'C': 1, 'gamma': 0.0001, 'kernel': 'linear'}

# to get the best model
print('The best model:', DT_grid_search.best_estimator_)

# the evaluation score of the best model
print('Evaluation score of the best model:', round(DT_grid_search.best_score_, 2))

# implement the model using the best value of the parameters
best_params = DT_grid_search.best_params_
best_DT_model = SVC(**best_params)
print('Best params: ', best_params)
print('Best model: ', best_DT_model)