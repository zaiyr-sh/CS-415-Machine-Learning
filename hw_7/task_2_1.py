import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("hw_7/lecture/Boston.csv")

# Prepare data for training models
y = data.pop('medv').values
X = data.values

# Hold-out validation

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# importing random forest classifier from ensemble module

from sklearn.ensemble import RandomForestClassifier

# implementing a RF classifier
RF1 = RandomForestClassifier(random_state = 1)

#RF1 model is using the default values for the hyper parameters
# n_estimators : (default=10)
# max_depth : (default=None) means that tree can grow as much as possible

# Training the model on the dataset
RF1.fit(X.astype('int'), y.astype('int'))

y_pred = RF1.predict(X)
print('Accuracy for RF1 model: %.2f' % accuracy_score(y.astype('int'), y_pred.astype('int')))
print("=====================================================================================")

# Hold-out validation
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 1, test_size = 0.20)

RF2 = RandomForestClassifier(random_state = 1)
# n_estimators : (default = 10)
# max_depth : (default = None) means that tree can grow as much as possible

# Training the model on the training dataset
RF2.fit(x_train.astype('int'), y_train.astype('int'))

y_train_pred = RF2.predict(x_train)
print('Training accuracy for RF2 model: %.2f' % accuracy_score(y_train.astype('int'), y_train_pred.astype('int')))

y_test_pred = RF2.predict(x_test)
print('Test accuracy for RF2 model: %.2f' % accuracy_score(y_test.astype('int'), y_test_pred.astype('int')))
print("=====================================================================================")

RF3 = RandomForestClassifier(n_estimators = 50, max_depth = 6, random_state = 1)
# n_estimators = 50, max_depth = 6, random_state = 1 81%
RF3.fit(x_train.astype('int'), y_train.astype('int'))

y_train_pred = RF3.predict(x_train)
print('Training accuracy for RF3 model: %.2f' % accuracy_score(y_train.astype('int'), y_train_pred.astype('int')))

y_test_pred = RF3.predict(x_test)
print('Test accuracy for RF3 model: %.2f' % accuracy_score(y_test.astype('int'), y_test_pred.astype('int')))
print("=====================================================================================")

RF4 = RandomForestClassifier(n_estimators = 15, max_depth = 5, random_state = 1)
# n_estimators = 15, max_depth = 5,random_state = 1 90%
RF4.fit(x_train.astype('int'), y_train.astype('int'))

y_train_pred = RF4.predict(x_train)
print('Training accuracy for RF4 model: %.2f' % accuracy_score(y_train.astype('int'), y_train_pred.astype('int')))

y_test_pred = RF4.predict(x_test)
print('Test accuracy for RF4 model: %.2f' % accuracy_score(y_test.astype('int'), y_test_pred.astype('int')))

# Use pickle to save our model so that we can use it later

import pickle

# Loading model
preTrainedDT = pickle.load(open('hw_7/lecture/regression_model.pkl','rb'))

print(round(preTrainedDT.predict([[1, 15, 3 , 1, 0.5, 6.5, 55, 3, 1, 200, 19, 33, 1]])[0], 1))