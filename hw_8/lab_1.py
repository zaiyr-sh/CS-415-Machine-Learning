import pandas as pd
from sklearn import model_selection

dataframe=pd.read_csv('hw_8/IRIS.csv')
print(dataframe.keys())

# Show the first five rows
dataframe.head()
dataframe.info()

array = dataframe.values # gets values except column headers
X = array[:,0:3]
y = array[:,4]

# split the data using Scikit-Learn's train_test_split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn import tree
modelDT = tree.DecisionTreeClassifier()

modelDT.fit(X_train, y_train)

y_pred_prob = modelDT.predict_proba(X_test)
y_pred = modelDT.predict(X_test)

print(modelDT)