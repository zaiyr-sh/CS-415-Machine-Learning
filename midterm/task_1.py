import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# load dataset as a panda data frame object
df = pd.read_csv('Boston.csv')

x=df[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']].values
y=df['medv'].values

# Implement the model
model = LinearRegression()

# Train (fit) the model
model.fit(x,y)

# Evaluate the model
r_sq = model.score(x,y)
print('Quality score of R squared value for the fitted model: ', round(r_sq, 2))

# Use the trained (fitted) model for prediction
xnew = np.array([[1, 18, 2.31, 1, 0.5, 6.5, 65, 4, 1, 250, 15, 333, 5]])
y_pred = model.predict(xnew)
print('Estimated Width of that fish: ', round(y_pred[0], 3))