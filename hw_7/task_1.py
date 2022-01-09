import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("hw_7/lecture/diabetesv2.csv")

# Prepare data for training models
y = data.pop('Outcome').values
X = data.values

# Hold-out validation

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)

# Use pickle to save our model so that we can use it later

import pickle

# Loading model
preTrainedDT = pickle.load(open('hw_7/lecture/classification_model.pkl','rb'))

print(preTrainedDT.predict([[0, 67, 62, 35, 1, 33.7, 0.5, 49]]))

y_pred = preTrainedDT.predict(x_test)
print('Training accuracy for model: %.3f' % accuracy_score(y_test, y_pred))