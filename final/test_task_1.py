import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data=pd.read_csv('final/tree_data.csv')


data.pop('id')
y = data.pop('tree_type').values
X = data.values

# split the data using Scikit-Learn's train_test_split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.80, test_size = 0.20, random_state = 1)

param_grid = {'C': [1, 5, 10, 100, 1000], 'gamma': [0.1, 1, 3, 5, 10],'kernel': ['rbf','linear', 'sigmoid']}


svr = SVC()
svr.fit(x_train, y_train)

predictions = svr.predict(x_test)
print(accuracy_score(y_test, predictions))

DT_grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=5,scoring="accuracy",return_train_score=True, verbose=3, n_jobs=-1) #If set to -1
DT_grid_search.fit(x_train, y_train)
# to get the best combination of hyperparameters
print(DT_grid_search.best_params_)
# to get the best model
print(DT_grid_search.best_estimator_)
# the evaluation score of the best model
print(DT_grid_search.best_score_)


# implement the model using the best value of the parameters
best_params = DT_grid_search.best_params_
best_DT_model=SVC(**best_params)

print(accuracy_score(y_test, predictions))
# score = svr.score(x_test, y_test)
# print("Support Vector Classifier: ", round(score, 2))

# lda = LinearDiscriminantAnalysis()
# lda.fit(x_train, y_train)
# score1 = lda.score(x_test, y_test)
# print("Linear Discriminant Analysis Classifier: ", round(score1, 2))

# rfc = RandomForestClassifier()
# rfc.fit(x_train, y_train)
# score2 = rfc.score(x_test, y_test)
# print("Random Forest Classifier: ", round(score2, 2))

# knc = KNeighborsClassifier()
# knc.fit(x_train, y_train)
# score3 = knc.score(x_test, y_test)
# print("K Neighbors Classifier: ", round(score3, 2))

# dtc = DecisionTreeClassifier()
# dtc.fit(x_train, y_train)
# score4 = dtc.score(x_test, y_test)
# print("Decision Tree Classifier: ", round(score4, 2))