# LEC_2
# Simple Linear regression 
import numpy as np

class myLinearRegression(object):
  def __init__(self, lrate = 0.01, niter = 10):
    self.lrate = lrate
    self.niter = niter

  def fit(self, X, y):
    # coefficients
    self.coefficient = np.zeros(1 + X.shape[1])

    # Errors
    self.errors = []

    # Cost function
    self.cost = []

    for i in range(self.niter):
      predicted = self.net_input(X)
      errors = y - predicted
      self.coefficient[1:] += self.lrate * X.T.dot(errors)
      self.coefficient[0] += self.lrate * errors.sum()
      cost = (errors**2).sum() / 2.0
      self.cost.append(cost)
    return self

  def net_input(self, X):
    """Compute net input"""
    return np.dot(X, self.coefficient[1:]) + self.coefficient[0]

  def prediction(self, X):
    """Compute linear prediction"""
    return self.net_input(X)


y = np.array([1, 2, 3, 4, 5])
X = np.array([[3], [5], [7], [9], [11]])

# learning lrate = 0.01
net = myLinearRegression(0.01, 10).fit(X,y)
print(net.coefficient[0])
print(net.coefficient[1])
print(net.net_input(4))
print(net.prediction(7))