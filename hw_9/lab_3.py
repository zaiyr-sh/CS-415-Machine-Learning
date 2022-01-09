# Experiment 2: Demonstration of the feature scaling for the classification task

import pandas as pd
import numpy as np
# load dataset as an panda data frame object
df = pd.read_csv('hw_9/diabetes.csv')

# data exploration
df.head()
df.dtypes
len(df.columns)
df.describe()

# Specify target and input features (variables)
X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
'BMI', 'DiabetesPedigreeFunction', 'Age' ]] # 'Outcome'

#x = df[features]
print(type(X)) # pandas.core.frame.DataFrame
print(X)
print(X.shape)

y = df['Outcome'].values
print(type(y)) # numpy.ndarray
print(y)
print(y.shape)

print("======================================================")

from sklearn.model_selection import train_test_split
# split the data into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 1)

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

print("======================================================")

from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier(n_neighbors = 5)

# Train the model using raw data
knc.fit(X_train, y_train)
# Evaluate the quality of the model
ACCURACY_rawdata = knc.score(X_test, y_test)
print('Accuracy square of trained model on raw data: ', ACCURACY_rawdata)

# Making prediction for new data
prediction = knc.predict([[0,67,62,35,1,33.7,0.5,49]])
print('Prediction for new data:', prediction)

# Train the model using data scaled with standardisation technique
knc.fit(X_train_scaled_standardization, y_train)
# Evaluate the quality of the model
ACCURACY_scaled_standardization = knc.score(X_test_scaled_standardization, y_test)
print('Accuracy square of trained model on standardized data: ', ACCURACY_scaled_standardization)

# Making prediction for new data
prediction = knc.predict(scalerX.transform([[0,67,62,35,1,33.7,0.5,49]]))
print('Prediction for new data with standardisation technique:', prediction)

# Train the model using data scaled with normalization technique
knc.fit(X_train_scaled_norm, y_train)
# Evaluate the quality of the model
ACCURACY_scaled_norm = knc.score(X_test_scaled_norm, y_test)
print('Accuracy square of trained model on normilized data: ', ACCURACY_scaled_norm)
# Making prediction for new data
prediction = knc.predict(normScalerX.transform([[0,67,62,35,1,33.7,0.5,49]]))
print('Prediction for new data with normalization technique:', prediction)

print("======================================================")

# Using data pipeline to automate training and scaling
from sklearn.pipeline import make_pipeline
model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors = 5))
model

model.fit(X_train, y_train)
predicted_target = model.predict(X_test)
predicted_target[:5]

predicted_target = model.predict([[0,67,62,35,1,33.7,0.5,49]])
predicted_target

score = model.score(X_test, y_test)
print(f"The accuracy of the model is {score:.3f} ")