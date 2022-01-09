# How to Save a Dataset as a File

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

# Load irisAll.csv files as a Pandas DataFrame
# https://www.kaggle.com/arshid/iris-flower-dataset
data = pd.read_csv("hw_7/IRIS.csv")

# Prepare data for training models
y = data.pop('species')
X = data

X.head()
y.head()

# Hold-out validation
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# saving the excel
x_train.to_excel('hw_7/lab_3/training_features_data.xlsx')

dframe = pd.read_excel('hw_7/lab_3/training_features_data.xlsx')
print(dframe)