# LEC_2
# Linear regression implementation 
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Train the model
y = np.array([1, 2, 3, 4, 5])
x = np.array([[3], [5], [7], [9], [11]])
model = LinearRegression()
model.fit(x, y)

# Evaluating the model quality
r_sq = model.score(x,y)
print('score of r square (quality): ', r_sq)
print("----------------------")

# Making prediction using the trained (fitted ) model

y_pred = model.predict(x)
print('predicted response:', y_pred)

y_pred = model.predict([[11]])
print('predicted response:', y_pred)

y_pred = model.predict([[11],[15]])
print('predicted response:', y_pred)

y_pred = model.predict([[13]])
print('predicted response:', y_pred)
print("----------------------")

# Making prediction
xnew = np.array([11])
xnew = xnew.reshape(-1, 1)
y_pred = model.predict(xnew)
print('predicted response:', y_pred)

xnew = np.array([11, 15])
xnew.shape
xnew = xnew.reshape(-1, 1)
xnew.shape
y_pred = model.predict(xnew)
print('predicted response:', y_pred)
print("----------------------")

# examining the fitted model coefficents

print('slope:', model.coef_)
print('intercept:', model.intercept_)

# You can notice that .intercept_ is a scalar, while .coef_ is an array.

# making prediction using the regression model
xnew = np.array([11])
y_pred = model.coef_ * xnew + model.intercept_
print('predicted response:', y_pred)