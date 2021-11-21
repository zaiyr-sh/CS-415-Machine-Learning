# Explore the Iris dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load irisAll.csv files as a Pandas DataFrame
# https://www.kaggle.com/arshid/iris-flower-dataset
data = pd.read_csv("hw_3/IRIS.csv")

# Explore the Iris dataset
sns.set_style('whitegrid')
sns.pairplot(data, hue='species')
plt.show()

# Prepare data for fitting model
X = data.iloc[:, [0, 1, 2, 3]].values

# Implementing and fitting the Kmeans model
KMmodel = KMeans(n_clusters = 3, random_state = 3)
KMmodel.fit(X)

def transformTextLabel(numericalLabel):
    if numericalLabel == 0:
        return 'setosa'
    if numericalLabel == 1:
        return 'versicolor'
    if numericalLabel == 2:
        return 'virginica'

predicted_clusters = KMmodel.labels_
data['predicted_clusters']=predicted_clusters
data['predictedTextLabel']=data['predicted_clusters'].apply(transformTextLabel)
# visualize the clusters using seaborn method 1
with sns.color_palette("hls", 8):
    sns.pairplot(data.iloc[:,[0,1,2,3,-1]], hue='predictedTextLabel')
plt.show()
# visualize the clusters using seaborn method 2
sns.set_style('whitegrid')
sns.pairplot(data.iloc[:,[0,1,2,3,-1]], hue='predictedTextLabel')
plt.show()

# -------------------------------------------------------------------------------------------------------------
# Confusion Matrix
confusion_matrix = pd.crosstab(data['species'], data['predictedTextLabel'])
print(confusion_matrix)

# -------------------------------------------------------------------------------------------------------------
# Visualizing the predicted_clusters with their centroids
plt.scatter(X[predicted_clusters == 0, 0], X[predicted_clusters == 0, 1], s = 100, c = 'purple', label = 'Iris-setosa')
plt.scatter(X[predicted_clusters == 1, 0], X[predicted_clusters == 1, 1], s = 100, c = 'orange', label = 'Iris-versicolour')
plt.scatter(X[predicted_clusters == 2, 0], X[predicted_clusters == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')
#Plotting the centroids of the predicted_clusters
plt.scatter(KMmodel.cluster_centers_[:, 0], KMmodel.cluster_centers_[:,1], s = 100, c = 'red', label = 'Centroids')
plt.legend()
plt.show()

# -------------------------------------------------------------------------------------------------------------
# Plotting predicted_clusters in 3D graph
from mpl_toolkits.mplot3d import Axes3D

predicted_clusters = KMmodel.labels_

fig = plt.figure(1, figsize=(7,7))
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
ax.scatter(X[:, 3], X[:, 0], X[:, 2],c=predicted_clusters.astype(np.float), edgecolor="k", s=50)
ax.set_xlabel("Petal width") # data.columns[3]
ax.set_ylabel("Sepal length") # data.columns[0]
ax.set_zlabel("Petal length") # data.columns[2]
plt.title("K Means", fontsize=14)
plt.show()

# -------------------------------------------------------------------------------------------------------------
# Plotting predicted clusters with their centroids in 3D graph
fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(1,1,1, projection='3d')
plt.scatter(X[predicted_clusters == 0, 0], X[predicted_clusters == 0, 1], s = 100, c = 'purple', label = 'Iris-setosa')
plt.scatter(X[predicted_clusters == 1, 0], X[predicted_clusters == 1, 1], s = 100, c = 'orange', label = 'Iris-versicolour')
plt.scatter(X[predicted_clusters == 2, 0], X[predicted_clusters == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')
#Plotting the centroids of the predicted_clusters
plt.scatter(KMmodel.cluster_centers_[:, 0], KMmodel.cluster_centers_[:,1], s = 100, c = 'red', label = 'Centroids')
plt.show()