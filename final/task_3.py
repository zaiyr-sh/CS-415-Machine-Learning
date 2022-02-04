import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("final/tree_data.csv")
data.pop("id")
Y = data.pop("tree_type").to_numpy()
X = data.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.80, test_size=0.20, random_state=1)

svm = SVC(random_state=1)
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)
svm_accuracy = accuracy_score(y_test, y_pred)
print(svm_accuracy)

lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)
y_pred = lda.predict(x_test)
lda_accuracy = accuracy_score(y_test, y_pred)
print(lda_accuracy)

rfc = RandomForestClassifier(random_state=1)
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)
rfc_accuracy = accuracy_score(y_test, y_pred)
print(rfc_accuracy)

knc = KNeighborsClassifier()
knc.fit(x_train, y_train)
y_pred = knc.predict(x_test)
knc_accuracy = accuracy_score(y_test, y_pred)
print(knc_accuracy)

dtc = DecisionTreeClassifier(random_state=1)
dtc.fit(x_train, y_train)
y_pred = dtc.predict(x_test)
dtc_accuracy = accuracy_score(y_test, y_pred)
print( dtc_accuracy)