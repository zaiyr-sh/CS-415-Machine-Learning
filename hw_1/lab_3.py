# LEC_2
# Simple Linear regression 
# Performing Gradient Descent for fining a local minimum
import numpy as np

X = np.array([1, 2, 3, 4, 5])
Y = np.array([3, 5, 7, 9, 11])

# Building the model y = ax + b
a = 0 # initial value of the coefficient a
b = 0 # initial value of the coefficient b

L = 0.01 # The learning Rate
epochs = 1000 # The number of iterations to perform gradient descent

n = float(len(X)) # Number of elements in X

# Performing Gradient Descent
for i in range(epochs):
  Y_pred = a * X + b # The current predicted value of Y
  D_a = (-2/n) * sum(X * (Y - Y_pred)) # Derivative wrt a
  D_b = (-2/n) * sum(Y - Y_pred) # Derivative wrt b
  a = a - L * D_a # Update a
  b = b - L * D_b # Update b

print ('a =',np.round(a), 'b =',np.round(b))