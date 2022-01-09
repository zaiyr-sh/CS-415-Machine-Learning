# Experiment I: Tuning the hyper parameters of RF model to classify the sonar data

""" Random Forest"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# summarize the sonar dataset

from pandas import read_csv

# load dataset
url = 'hw_6/sonar.csv'
df = read_csv(url, header = None)
df.head()

df.info()
#the classification of sonar signals
#The label associated with each record contains the letter "R" if the object
#is a rock and "M" if it is a mine (metal cylinder)
df[60].value_counts()

df[60].value_counts().plot.barh();
plt.show()
df.describe().T
data = df.values

# preparing the input and output variables
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)

# importing random forest classifier from ensemble module

from sklearn.ensemble import RandomForestClassifier

# implementing a RF classifier
RF1 = RandomForestClassifier(random_state = 1)

#RF1 model is using the default values for the hyper parameters
# n_estimators : (default=10)
# max_depth : (default=None) means that tree can grow as much as possible

# Training the model on the dataset
RF1.fit(X, y)

y_pred = RF1.predict(X)
print('Accuracy for RF1 model: %.2f' % accuracy_score(y, y_pred))
# >>> Accuracy for RF1 model: 1.00
# This result looks overfitting (memorization of the dataset) of the machine learning
# model. To be sure you need to estimate the accuracy of the trained model using test
# data set.
print("=====================================================================================")

# Hold-out validation
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 1, test_size = 0.20)

RF2 = RandomForestClassifier(random_state = 1)
# n_estimators : (default = 10)
# max_depth : (default = None) means that tree can grow as much as possible

# Training the model on the training dataset
RF2.fit(x_train, y_train)

y_train_pred = RF2.predict(x_train)
print('Training accuracy for RF2 model: %.2f' % accuracy_score(y_train, y_train_pred))

y_test_pred = RF2.predict(x_test)
print('Test accuracy for RF2 model: %.2f' % accuracy_score(y_test, y_test_pred))
# >>> Training accuracy for RF2 model: 1.00
# >>> Test accuracy for RF2 model: 0.76
# Unbiased estimated accuracy of the trained model is 76%, which is computed based on the test data
# set. As seen in this simulation, a huge difference between the training and test accuracy is a typical
# indication of the overfitting (memorization of the dataset) of the machine learning model. Model
# parameters are required to be set and tuned carefully.
print("=====================================================================================")

RF3 = RandomForestClassifier(n_estimators = 50, max_depth = 6, random_state = 1)
# n_estimators = 50, max_depth = 6, random_state = 1 81%
RF3.fit(x_train, y_train)

y_train_pred = RF3.predict(x_train)
print('Training accuracy for RF3 model: %.2f' % accuracy_score(y_train, y_train_pred))

y_test_pred = RF3.predict(x_test)
print('Test accuracy for RF3 model: %.2f' % accuracy_score(y_test, y_test_pred))
# >>> Training accuracy for RF3 model: 1
# >>> Test accuracy for RF3 model: 0.81
# When we set the parameter of the n_estimators and max_depth to the corresponding values of 50
# and 6. Unbiased estimated accuracy of the trained model is improved and become 81%. This model
# is better than the previous one but still there is indication of the overfitting.
print("=====================================================================================")

RF4 = RandomForestClassifier(n_estimators = 15, max_depth = 5, random_state = 1)
# n_estimators = 15, max_depth = 5,random_state = 1 90%
RF4.fit(x_train, y_train)

y_train_pred = RF4.predict(x_train)
print('Training accuracy for RF4 model: %.2f' % accuracy_score(y_train, y_train_pred))

y_test_pred = RF4.predict(x_test)
print('Test accuracy for RF4 model: %.2f' % accuracy_score(y_test, y_test_pred))
# >>> Training accuracy for RF4 model: 0.99
# >>> Test accuracy for RF4 model: 0.90
# When we have tuned the parameter of the n_estimators and max_depth to the corresponding
# values of 15 and 5. Unbiased estimated accuracy of the trained model is improved and become 90%.
# This model is better than the previous one.