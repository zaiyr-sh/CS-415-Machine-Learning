# KMeans implementation in sciLearn

import pandas as pd

# Using K-means clustering on Iris dataset:
from sklearn.cluster import KMeans

# Load irisAll.csv files as a Pandas DataFrame
# https://www.kaggle.com/arshid/iris-flower-dataset
data = pd.read_csv("hw_3/IRIS.csv")

# Prepare data for fitting model
X = data.iloc[:, [0, 1, 2, 3]].values

# Implementing and fitting the Kmeans model
model = KMeans(n_clusters = 3, random_state = 3)
model.fit(X)

# Using the fitted model to predict
predicted_clusters = model.predict(X)
## or fit & predict methods together
predicted_clusters = model.fit_predict(X)
## or using labels_ method attribute
predicted_clusters = model.labels_

# Accuracy of the model
# Converting text labels to numerical labels
labels = data.iloc[:, 4].values
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
numerical_labels = encoder.fit_transform(labels)

from sklearn.metrics import accuracy_score
predicted_clusters = model.predict(X)
print(accuracy_score(numerical_labels, predicted_clusters))
KM_model_accuracy = accuracy_score(numerical_labels, predicted_clusters)
print('K = 3 KMeans accuracy: {0:.4f}%'.format(KM_model_accuracy * 100))
print("----------------------")

# Implementing and training the log regression model
from sklearn.linear_model import LogisticRegression
LogReg = LogisticRegression(random_state = 3)
LogReg.fit(X, labels)

# Accuracy of the model
from sklearn.metrics import accuracy_score
predictedClass = LogReg.predict(X)
print(accuracy_score(labels, predictedClass))
LogReg_model_accuracy = accuracy_score(labels, predictedClass)
print('LogReg accuracy: {0:.4f}%'.format(LogReg_model_accuracy * 100))
print("----------------------")

# Use the trained (fitted) model for prediction
xnew = X[149,:].reshape(1, -1) # the last row of data in the iris dataset
#xnew = np.array([[5.9, 3. , 5.1, 1.8]])
#xnew = X[148:150,:] # the last two rows of data in the iris dataset
#xnew = np.array([[5.9, 3. , 5.1, 1.8]])
y_pred = model.predict(xnew)
print('predicted response:', y_pred)