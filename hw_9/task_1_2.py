import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Pretty display for notebooks

# Load the Boston housing dataset
data = pd.read_csv('hw_9/data.csv')
# features = data.iloc[:, 2:].columns.tolist()
data.drop(['Unnamed: 32','id'],axis=1,inplace=True)
data.diagnosis=[1 if each=="M" else 0 for each in data.diagnosis]
# X = data[["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]].values
X = data.loc[:,data.columns!="diagnosis"]

# y = data.iloc[1]

y=data.loc[:,"diagnosis"].values

# print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# data scaling using standardization with sklearn
from sklearn.preprocessing import StandardScaler
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

from sklearn.svm import SVC
# train the model on train set
svc = SVC(C=0.1, gamma=0.0001, kernel='linear')  

svc.fit(X_train, y_train)

# Evaluate the quality of the model
r_sq_rawdata = svc.score(X_test, y_test)
print('score of r square trained on raw data: ', r_sq_rawdata)
# Making prediction for new data
# prediction = svc.predict([[1, 15, 3 , 1, 0.5, 6.5, 55, 3, 1, 200, 19, 33, 1]])
# print(prediction)
# Train the model using data scaled with standardisation technique

yy = y_train_scaled_standardization.ravel()
train_y = np.array(yy).astype(int)

svc.fit(X_train_scaled_standardization, train_y)
# print(X_train_scaled_standardization)
# Evaluate the quality of the model
r_sq_scaled_standardization = svc.score(X_test_scaled_standardization.round(), y_test_scaled_standardization.round())
print('score of r square trained on standardized data: ', r_sq_scaled_standardization)
# Making prediction for new data
# prediction = svc.predict(scalerX.transform([[1, 15, 3 , 1, 0.5, 6.5, 55, 3, 1, 200, 19, 33, 1]])) 
# y_new_inverse = normScalery.inverse_transform(prediction.reshape(-1, 1))
# print(y_new_inverse)
# Train the model using data scaled with normalization technique

yyy = y_train_scaled_norm.ravel()
train_yy = np.array(yyy).astype(int)

svc.fit(X_train_scaled_norm, train_yy)
# Evaluate the quality of the model
r_sq_scaled_norm = svc.score(X_test_scaled_norm,y_test_scaled_norm)
print('score of r square trained on normalized data: ', r_sq_scaled_norm)
# Making prediction for new data
# prediction = svc.predict(normScalerX.transform([[1, 15, 3 , 1, 0.5, 6.5, 55, 3, 1, 200, 19, 33, 1]]))
# y_new_inverse = scalery.inverse_transform(prediction.reshape(-1, 1))
# print(y_new_inverse)