# How to train ML model and save them to disk

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

# Load irisAll.csv files as a Pandas DataFrame
# https://www.kaggle.com/arshid/iris-flower-dataset
data = pd.read_csv("hw_7/IRIS.csv")

# Some information about dataset
print (data.shape)
print(type(data)) # 'pandas.core.frame.DataFrame'
data.dtypes

data.dtypes
data.head()
data.describe().T

# Prepare data for training models
y = data.pop('species').values
X = data.values

# X.head().values
# y.head().values

# Hold-out validation

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# Training the decision tree (DT) model
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

print(dt.__dict__)

y_pred = dt.predict(x_test)
print(accuracy_score(y_test, y_pred))

# Use pickle to save our model so that we can use it later

import pickle

# Saving model
pickle.dump(dt, open('hw_7/lab_1/model.pkl','wb'))

# Loading model
preTrainedDT = pickle.load(open('hw_7/lab_1/model.pkl','rb'))

print(dt)
print(preTrainedDT.predict([[1, 1, 4, 4]]))