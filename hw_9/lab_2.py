# Experiment 1: Demonstration of the feature scaling for the regression task

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# load dataset as an panda data frame object
df = pd.read_csv('hw_9/Boston.csv')
# Explore the data

print('\nDataFrame Shape :', df.shape)
print('\nNumber of rows :', df.shape[0])
print('\nNumber of columns :', df.shape[1])
df.head()
df.dtypes
len(df.columns)

# Specify target and input features (variables)
target = df.iloc[:, 14].name
target
features = df.iloc[:, 1:14].columns.tolist()
features
X = df[features]
#x = df[features]
print(type(X)) # pandas.core.frame.DataFrame
print(X)
print(X.shape)

y=df[target].values
print(type(y)) # numpy.ndarray
print(y)
print(y.shape)

print("======================================================")

# split the data into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 1)

# data scaling using standardization with sklearn
from sklearn.preprocessing import StandardScaler
# use the training parameters and re-use them to scale the test dataset.
# If we standardize our training dataset, we need to keep the parameters
# (mean and standard deviation for each feature). Then, we’d use these parameters
# to transform training data, test data and any future data later on

# fit scaler on training data using standardization
scalerX = StandardScaler().fit(X_train)
scalery = StandardScaler().fit(y_train.reshape(-1, 1))

# transform (scale) the data
X_train_scaled_standardization = scalerX.transform(X_train)
y_train_scaled_standardization = scalery.transform(y_train.reshape(-1, 1))
X_test_scaled_standardization = scalerX.transform(X_test)
y_test_scaled_standardization = scalery.transform(y_test.reshape(-1, 1))

# data scaling using normalization with sklearn
from sklearn.preprocessing import MinMaxScaler

# fit scaler on training data using normalization
normScalerX = MinMaxScaler().fit(X_train)
normScalery = MinMaxScaler().fit(y_train.reshape(-1, 1))

# transform (scale) the data
X_train_scaled_norm = normScalerX.transform(X_train)
y_train_scaled_norm = normScalery.transform(y_train.reshape(-1, 1))
X_test_scaled_norm = normScalerX.transform(X_test)
y_test_scaled_norm = normScalery.transform(y_test.reshape(-1, 1))

print("======================================================")

# Implementation of the ML regression model
from sklearn import svm
svr= svm.SVR(kernel = 'rbf', C = 100)

# Train the model using raw data
svr.fit(X_train, y_train)
# Evaluate the quality of the model
r_sq_rawdata = svr.score(X_test, y_test)
print('Score of r square trained on raw data: ', r_sq_rawdata)
# Making prediction for new data
prediction=svr.predict([[1, 15, 3 , 1, 0.5, 6.5, 55, 3, 1, 200, 19, 33, 1]])
print('Prediction for new data:', prediction)

# Train the model using data scaled with standardisation technique
svr.fit(X_train_scaled_standardization, y_train_scaled_standardization)
# Evaluate the quality of the model
r_sq_scaled_standardization = svr.score(X_test_scaled_standardization, y_test_scaled_standardization)
print('score of r square trained on standardized data: ', r_sq_scaled_standardization)
# Making prediction for new data
prediction=svr.predict(scalerX.transform([[1, 15, 3 , 1, 0.5, 6.5, 55, 3, 1, 200, 19, 33, 1]]))
y_new_inverse = normScalery.inverse_transform(prediction.reshape(-1, 1))
print('Prediction for new data with standardisation technique:', y_new_inverse)

# Train the model using data scaled with normalization technique
svr.fit(X_train_scaled_norm, y_train_scaled_norm)
# Evaluate the quality of the model
r_sq_scaled_norm = svr.score(X_test_scaled_norm,y_test_scaled_norm)
print('score of r square trained on normalized data: ', r_sq_scaled_norm)
# Making prediction for new data
prediction=svr.predict(normScalerX.transform([[1, 15, 3 , 1, 0.5, 6.5, 55, 3, 1, 200,
19, 33, 1]]))
y_new_inverse = scalery.inverse_transform(prediction.reshape(-1, 1))
print('Prediction for new data with normalization technique:', y_new_inverse)

print("======================================================")

# Implementation of the ML regression model
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor(n_neighbors = 7)

# Train the model using raw data
knr.fit(X_train,y_train)
# Evaluate the quality of the model
r_sq_rawdata = knr.score(X_test, y_test)
print('Score of r square trained on raw data: ', r_sq_rawdata)

# Train the model using data scaled with standardisation technique
knr.fit(X_train_scaled_standardization, y_train_scaled_standardization)
# Evaluate the quality of the model
r_sq_scaled_standardization = knr.score(X_test_scaled_standardization, y_test_scaled_standardization)
print('Score of r square trained on standardized data: ', r_sq_scaled_standardization)

# Train the model using data scaled with normalization technique
knr.fit(X_train_scaled_norm, y_train_scaled_norm)
# Evaluate the quality of the model
r_sq_scaled_norm = knr.score(X_test_scaled_norm, y_test_scaled_norm)
print('Score of r square trained on normalized data: ', r_sq_scaled_norm)

print("======================================================")

# Implementation of the ML ensemble regression model
from sklearn import ensemble
xtr = ensemble.ExtraTreesRegressor(n_estimators = 10, random_state = 1)

# Train the model using raw data
xtr.fit(X_train, y_train)
# Evaluate the quality of the model
r_sq_rawdata = xtr.score(X_test, y_test)
print('Score of r square trained on raw data: ', r_sq_rawdata)

# Train the model using data scaled with standardisation technique
xtr.fit(X_train_scaled_standardization, y_train_scaled_standardization)
# Evaluate the quality of the model
r_sq_scaled_standardization = xtr.score(X_test_scaled_standardization, y_test_scaled_standardization)
print('Score of r square trained on standardized data: ', r_sq_scaled_standardization)

# Train the model using data scaled with normalization technique
xtr.fit(X_train_scaled_norm, y_train_scaled_norm)
# Evaluate the quality of the model
r_sq_scaled_norm = xtr.score(X_test_scaled_norm, y_test_scaled_norm)
print('Score of r square trained on normalized data: ', r_sq_scaled_norm)