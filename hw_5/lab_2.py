# Experiment I: Using the SVM model to classify the iris flowers

# Loading the training data and some exploration for the data
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Load irisAll.csv files as a Pandas DataFrame
data = pd.read_csv("hw_5/IRIS.csv")

# Some information about dataset
print (data.shape)
print(type(data))
data.dtypes

data.dtypes
data.head()
data.describe()

# Prepare data for training models
labels = data.pop('species')
train = data

train.head()
labels.head()

# Implement the SVM model
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)

# Fit the model for the data
classifier.fit(train, labels)

# Make the prediction
y_pred = classifier.predict(train)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(labels, y_pred)
print(cm)
print(accuracy_score(labels, y_pred))