import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Pretty display for notebooks

# Load the Boston housing dataset
data = pd.read_csv('hw_7/lecture/Boston.csv')
x = data[["crim","zn","indus","chas","nox","rm","age","dis","rad","tax","ptratio","black","lstat"]].values


y=data["medv"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

dt = LinearRegression()
dt.fit(x_train, y_train)


r_sq = dt.score(x,y)
print('score of r square: ', r_sq)

y_pred = dt.predict(x_test)


# Use pickle to save our model so that we can use it later
import pickle

# Saving model
pickle.dump(dt, open('hw_7/regression_model.pkl','wb'))

# Loading model
preTrainedDT=pickle.load(open('hw_7/regression_model.pkl','rb'))
# result = preTrainedDT.score(x_test, y_test)
# print(result)

print(preTrainedDT.predict([[ 1, 15, 3 , 1, 0.5, 6.5, 55, 3, 1, 200, 19, 33, 1]]))

y_test_pred = dt.predict(x_test)
# print(dt.score(y_test, y_test_pred))
from sklearn.metrics import r2_score
print(r2_score(y_test,y_test_pred))