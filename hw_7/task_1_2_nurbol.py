import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Classification
model = pickle.load(open("hw_7/lecture/classification_model.pkl", "rb"))
predicted = model.predict([[0, 67, 62, 35, 1, 33.7, 0.5, 49]])
print("Predicted: ", predicted)

data = pd.read_csv("hw_7/lecture/diabetesv2.csv")
Y = data.pop("Outcome").to_numpy()
X = data.to_numpy()
print(X)

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=1, test_size=0.20)
y_pred = model.predict(x_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))

# Regression
model = pickle.load(open("hw_7/lecture/regression_model.pkl", "rb"))
predicted = model.predict([[1, 15, 3, 1, 0.5, 6.5, 55, 3, 1, 200, 19, 33, 1]])
print("Predicted regression: ", *predicted)

data = pd.read_csv("hw_7/lecture/Boston.csv")
Y = data.pop("medv").to_numpy()
X = data.iloc[:, 1:15].to_numpy()
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=1, test_size=0.20)
print(len(x_test))
print(len(y_test))
r_2 = model.score(x_test, y_test)
print("R square: ", r_2)
