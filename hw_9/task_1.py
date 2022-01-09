import pandas as pd
import numpy as np
from sklearn.svm import SVC

data = pd.read_csv('hw_9/data.csv')

data.pop('id')
y = data.pop('diagnosis').values
X = data.values

X = np.nan_to_num(X.astype(np.float64))

# split the data using Scikit-Learn's train_test_split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75, test_size = 0.25, random_state = 1)

# data scaling using standardization with sklearn
from sklearn.preprocessing import StandardScaler
# fit scaler on training data using standardization
scalerX = StandardScaler().fit(X_train)
# transform (scale) the data
X_train_scaled_standardization = scalerX.transform(X_train)
X_test_scaled_standardization = scalerX.transform(X_test)

# data scaling using normalization with sklearn
from sklearn.preprocessing import MinMaxScaler
# fit scaler on training data using normalization
normScalerX = MinMaxScaler().fit(X_train)
# transform (scale) the data
X_train_scaled_norm = normScalerX.transform(X_train)
X_test_scaled_norm = normScalerX.transform(X_test)

svr = SVC(C = 0.1, gamma = 0.0001, kernel = 'linear')
# Train the model using raw data
svr.fit(X_train, y_train)
# Evaluate the quality of the model
ACCURACY_rawdata = svr.score(X_test, y_test)
print('Accuracy square of trained model on raw data: ', round(ACCURACY_rawdata, 3))

# Train the model using data scaled with standardisation technique
svr.fit(X_train_scaled_standardization, y_train)
# Evaluate the quality of the model
ACCURACY_scaled_standardization = svr.score(X_test_scaled_standardization, y_test)
print('Accuracy square of trained model on standardized data: ', round(ACCURACY_scaled_standardization, 3))

# Train the model using data scaled with normalization technique
svr.fit(X_train_scaled_norm, y_train)
# Evaluate the quality of the model
ACCURACY_scaled_norm = svr.score(X_test_scaled_norm, y_test)
print('Accuracy square of trained model on normilized data: ', round(ACCURACY_scaled_norm, 3))