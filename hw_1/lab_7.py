# LEC_2
# Linear regression implementation in sciLearn
# Multiple Linear Regression With scikit-learn

import numpy as np
from sklearn.linear_model import LinearRegression

# x1 = [3, 5, 3, 3, 2, 4]
# x2 = [1, 2, 2, 5, 3, 5]
#  y = [8, 12, 10, 16, 11, ?]
# load dataset as python list object
x = [[3, 1], [5, 2], [3, 2], [3, 5], [2, 3]] # list
y = [8, 12, 10, 16, 11] # list

print(type(x))
print(np.shape(x))
print(x)
print(y)
print("----------------------")

# optionally you can convert to numpy arrays
# x, y = np.array(x), np.array(y)

#print(type(x))
#print(np.shape(x))
#print(x)
#print(y)

# Implement the regression model and fit it (train it)
model = LinearRegression().fit(x, y)

# Evaluate the quality of the model
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)

# Display the learned parameters of the model
print('intercept:', model.intercept_)
print('slope:', model.coef_)

# Use the trained (fitted) model for prediction
y_pred = model.predict([[4,5]])
print('predicted response:', y_pred)