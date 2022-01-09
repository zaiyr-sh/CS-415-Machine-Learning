# Experiment I: Using the RF model to classify the iris flowers

# Loading the training data and some exploration for the data
import pandas as pd

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

# importing random forest classifier from ensemble module
from sklearn.ensemble import RandomForestClassifier

# implementing a RF classifier
RF = RandomForestClassifier(n_estimators = 100)

# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
RF.fit(train, labels)

# performing predictions on the test dataset
y_pred = RF.predict(train)

# metrics are used to find accuracy or error
from sklearn import metrics
print()

# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(labels, y_pred))

cm = metrics.confusion_matrix(labels, y_pred)
print(cm)