import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


df = pd.read_csv('hw_7/lecture/diabetesv2.csv')

print(df.head())
x = df[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]].values


y=df["Outcome"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

from sklearn.tree import DecisionTreeClassifier
# dt = DecisionTreeClassifier()
# dt.fit(x_train, y_train)


# print(dt.dict)
# y_pred = dt.predict(x_test)
# print(accuracy_score(y_test, y_pred))
#Use pickle to save our model so that we can use it later
import pickle

# Saving model
# pickle.dump(dt, open('classification_model.pkl','wb'))

# # Loading model
# preTrainedDT=pickle.load(open('classification_model.pkl','rb'))
# # result = preTrainedDT.score(x_test, y_test)
# # print(result)

# print(preTrainedDT.predict([[ 0, 67, 62, 35, 1, 33.7, 0.5, 49]]))

# y_test_pred = dt.predict(x_test)
# print(accuracy_score(y_test, y_test_pred))


# Loading model
preTrainedDT = pickle.load(open('hw_7/lecture/classification_model.pkl','rb'))

print(preTrainedDT.predict([[0, 67, 62, 35, 1, 33.7, 0.5, 49]]))

y_train_pred = preTrainedDT.predict(x_test)
print('Training accuracy for model: %.2f' % accuracy_score(y_test, y_train_pred))