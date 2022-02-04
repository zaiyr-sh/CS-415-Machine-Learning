# Loading data and some data explatory tasks 
import pandas as pd
from sklearn.svm import SVR

data = pd.read_csv("hw_6/Real estate.csv")

# Prepare data for training models

x = data.drop(['No', 'X1 transaction date', 'Y house price of unit area'], axis=1)
y = data['Y house price of unit area']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1, test_size = 0.20)

svr = SVR()
svr.fit(x_train, y_train)

# y_pred = svr.predict(x_test)

score = svr.score(x_test, y_test)
print(round(score, 2))