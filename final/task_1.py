import pandas as pd
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('final/tree_data.csv')

data.pop('id')
y = data.pop('tree_type').values
X = data.values

# split the data using Scikit-Learn's train_test_split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.80, test_size = 0.20, random_state = 1)

svr = SVC()
svr.fit(x_train, y_train)
score = svr.score(x_test, y_test)
print("Support Vector Classifier: ", round(score, 2))


lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)
score1 = lda.score(x_test, y_test)
print("Linear Discriminant Analysis Classifier: ", round(score1, 2))

rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
score2 = rfc.score(x_test, y_test)
print("Random Forest Classifier: ", round(score2, 2))

knc = KNeighborsClassifier()
knc.fit(x_train, y_train)
score3 = knc.score(x_test, y_test)
print("K Neighbors Classifier: ", round(score3, 2))

dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
score4 = dtc.score(x_test, y_test)
print("Decision Tree Classifier: ", round(score4, 2))
