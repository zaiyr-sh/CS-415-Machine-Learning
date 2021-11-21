# Simple Clustering Example
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
    'x1': [23,33,21,27,33,23,41,22,35,34,67,54,57,43,50,57,59,52,65,47,49,48,35,33,41,44,36],
    'x2': [73,51,55,75,59,71,73,57,73,75,51,32,40,47,53,36,35,58,59,50,25,20,14,11,20,5,31] 
}

plt.scatter(data['x1'], data['x2'])
plt.xlabel("Input 1: X1")
plt.ylabel("Input 2: X2")
plt.show()

X = DataFrame(data,columns=['x1','x2'])

KMmodel = KMeans(n_clusters = 3)
KMmodel.fit(X)

centroids = KMmodel.cluster_centers_
print(centroids)

plt.scatter(data['x1'], data['x2'], c = KMmodel.labels_.astype(float))
plt.scatter(centroids[:, 0], centroids[:, 1], c = 'red', s = 50)
plt.xlabel("Input 1: X1")
plt.ylabel("Input 2: X2")
plt.show()

# Use the trained (fitted) model for prediction
xnew = np.array([[21, 71]])
y_pred = KMmodel.predict(xnew)
print('predicted response:', y_pred)