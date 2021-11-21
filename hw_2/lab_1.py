# LEC_3
# Developing Logistic regression classification 
# model to predict the flower types
# With scikit-learn

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Load irisAll.csv files as a Pandas DataFrame
# https://www.kaggle.com/arshid/iris-flower-dataset 
data = pd.read_csv("hw_2/IRIS.csv")

# Some information about dataset
print (data.shape)
print(type(data)) # 'pandas.core.frame.DataFrame' data.dtypes
print("----------------------")

data.dtypes 
data.head() 
data.describe()

# Prepare input and output data 
# for training models
y = data.pop('species')
X = data

X.head()
y.head()

# Implementing the Logistic regression model
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 
model = LogisticRegression()

# Training the Log regression model
model.fit(X, y)

# Measuring the quality of trained model
y_predForTrainingData = model.predict(X) 
print(accuracy_score(y, y_predForTrainingData))
print("----------------------")

# Scoring the Trained Log Reg model
# (Using the trained model for prediction)
new_input=[[3,4,5,4]] # predict function required 2D array input. 
predictions = model.predict(new_input)
print(predictions)
