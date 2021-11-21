# LEC_2
# Linear regression implementation in sciLearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# load our input data as a numpy array object
x = np.array([1, 2, 3, 4, 5])
# or
# load our input data as a panda dataframe object
# x = pd.DataFrame({'inputs': [1, 2, 3, 4, 5]})

# Examine the data
print(type(x))
print(x)
print(x.shape)
print("----------------------")
# x = [[1], [2], [3], [4], [5]] # as list

y = np.array([3, 5, 7, 9, 11])

print(type(y))
print(y)
print(y.shape)
print("----------------------")

# №1 Modify the input data shape by "reshaping" input data
x = x.reshape(-1, 1)
print(x.shape) # (5, 1)
print(x)
print("----------------------")

# №2 load our data as an numpy array object
x = np.array([[1], [2], [3], [4], [5]])
print(type(x))
print(x)
print(x.shape)
# or
# №3 load our data as a panda dataframe object
x = pd.DataFrame({'inputs': [[1], [2], [3], [4], [5]]})
print(type(x))
print(x)
print(x['inputs'])
print(x.shape)
print("------------------------------------------------------------------")