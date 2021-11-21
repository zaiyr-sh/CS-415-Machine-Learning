# Developing Logistic regression classification 
# model to predict the flower types
# With scikit-learn

import pandas as pd

# Load Fish.csv files as a Pandas DataFrame
data = pd.read_csv("hw_2/Fish.csv")

# Some information about dataset
print (data.shape)
print(type(data)) # 'pandas.core.frame.DataFrame' data.dtypes
print("----------------------")

data.dtypes 
data.head() 
data.describe()

# Prepare input and output data 
# for training models
y = data.pop('Species')
X = data

X.head()
y.head()

# Implementing the Logistic regression model
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 
model = LogisticRegression(solver = 'lbfgs', max_iter = 10000)

# Training the Log regression model
model.fit(X, y)

# Measuring the quality of trained model
y_predForTrainingData = model.predict(X) 
print(round(accuracy_score(y, y_predForTrainingData), 2))
print("----------------------")

# Scoring the Trained Log Reg model
# (Using the trained model for prediction)
new_input=[[241, 23, 25, 30, 11, 4]] # predict function required 2D array input. 
predictions = model.predict(new_input)
print(predictions)
