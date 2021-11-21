# KMeans implementation in sciLearn

import pandas as pd
from pandas.core.frame import DataFrame

# Using K-means clustering on Iris dataset:
from sklearn.cluster import KMeans

# Load Fish.csv files as a Pandas DataFrame
data = pd.read_csv("hw_3/Fish.csv")
dfBream = data.loc[data['Species'] == 'Bream'] 
dfWhiteFish = data.loc[data['Species'] == 'Whitefish']
data = pd.concat([dfBream, dfWhiteFish])

# Prepare data for fitting model
X = data[['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']].values

# Implementing and fitting the Kmeans model
model = KMeans(n_clusters = 2, random_state = 5)
model.fit(X)

# Using the fitted model to predict
predicted_clusters = model.predict(X)

# Accuracy of the model
# Converting text labels to numerical labels
labels = data.pop('Species')
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
numerical_labels = encoder.fit_transform(labels)

from sklearn.metrics import accuracy_score
predicted_clusters = model.predict(X)
print(round(accuracy_score(numerical_labels, predicted_clusters), 2))
KM_model_accuracy = accuracy_score(numerical_labels, predicted_clusters)
print('K = 2 KMeans accuracy: {0:.4f}%'.format(KM_model_accuracy * 100))

# Implementing and training the log regression model
from sklearn.linear_model import LogisticRegression
LogReg = LogisticRegression(random_state = 5, solver = 'lbfgs', max_iter = 10000)
LogReg.fit(X, labels)

# Use the trained (fitted) model for prediction
xnew = [[270, 23, 25, 28, 8, 4]]

y_pred = model.predict(xnew)
print('predicted response:', y_pred)