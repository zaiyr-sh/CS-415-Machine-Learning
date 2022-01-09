import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("hw_7/lecture/Boston.csv")

# Prepare data for training models
data.pop('id')
y = data.pop('medv').values
X = data.values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)

# Use pickle to save our model so that we can use it later

import pickle

# Loading model
preTrainedDT = pickle.load(open('hw_7/lecture/regression_model.pkl','rb'))

print(round(preTrainedDT.predict([[1, 15, 3 , 1, 0.5, 6.5, 55, 3, 1, 200, 19, 33, 1]])[0], 1))

r_2 = preTrainedDT.score(x_test, y_test)
print('Training accuracy for model: %.3f' % r_2)