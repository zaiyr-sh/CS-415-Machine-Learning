import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def grid_search(name):
    param_grid = {
        'C': [1, 5, 10, 100, 1000],
        'gamma': [0.1, 1, 3, 5, 10],
        'kernel': ['rbf', 'linear', 'sigmoid']
    }

    DT_grid_search = GridSearchCV(estimator = name,
                                param_grid = param_grid,
                                cv = 3,
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

data = pd.read_csv('final/tree_data.csv')

data.pop('id')
y = data.pop('tree_type').values
X = data.values

# split the data using Scikit-Learn's train_test_split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.80, test_size = 0.20, random_state = 1)

svr = SVC()
svr.fit(x_train, y_train)
score = svr.score(x_test, y_test)
print("Support Vector Classifier: ", round(score, 2))
grid_search(svr)
