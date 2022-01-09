# Experiment II: illustrate the Hold-out and K-fold cross validation

# Loading data and some data explatory tasks 
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Load irisAll.csv files as a Pandas DataFrame
# https://www.kaggle.com/arshid/iris-flower-dataset
data = pd.read_csv("hw_6/IRIS.csv")

# Some information about dataset
print (data.shape)
print(type(data)) # 'pandas.core.frame.DataFrame'
data.dtypes

data.dtypes
data.head()
data.describe()

### Training the model using whole dataset 
print("\nTraining the model using whole dataset: ")

# Prepare data for training models
labels = data.pop('species')
train=data

train.head()
labels.head()

# Training the decision tree (DT) model
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
dt1 = DecisionTreeClassifier()
dt1.fit(train, labels)

y_predForTrainingData = dt1.predict(train)
print(accuracy_score(labels, y_predForTrainingData))

### Hold-out validation
print("\nHold-out validation: ")

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train, labels, test_size = 0.20) # 20% for testing

dt2 = DecisionTreeClassifier()
dt2.fit(x_train, y_train)

y_pred = dt2.predict(x_test)
print(accuracy_score(y_test, y_pred))

### k-Fold Cross-Validation
print("\nk-Fold Cross-Validation: ")

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# prepare the cross-validation procedure
cv = KFold(n_splits = 10, random_state=1, shuffle=True)

# Design decision tree model
dt3 = DecisionTreeClassifier()

# evaluate model
scores = cross_val_score(dt3, train, labels, scoring='accuracy', cv=cv, n_jobs = -1)

# report performance
from numpy import mean,std
print('Accuracy: %.3f, (%.3f) stdDev' % (mean(scores), std(scores)))

#stratified k-fold cross-validation
print("\nstratified k-fold cross-validation: ")

from sklearn.model_selection import StratifiedKFold
scv = StratifiedKFold(n_splits=5, shuffle=True, random_state = 1)

# build model
dt4 = DecisionTreeClassifier()

# evaluate model
scores2 = cross_val_score(dt4, train, labels, scoring='accuracy', cv=scv, n_jobs=-1)

# report performance
from numpy import mean,std
print('Accuracy: %.3f, (%.3f) stdDev' % (mean(scores2), std(scores2)))