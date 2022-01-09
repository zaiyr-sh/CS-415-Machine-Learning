import pandas as pd

df = pd.read_csv('midterm/diabetes.csv')
print(df)
# setting  input
x = df[['Pregnancies','Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]

# #setting output
y=df[['Outcome']]
x.head()
y.head()


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model=LogisticRegression()
# Training the Log regression model
model.fit(x, y)

y_predForTrainingData = model.predict(x)
print(accuracy_score(y, y_predForTrainingData))

new_input=[[1,148,72,35,0,33.66,0.627,50]] 
predictions = model.predict(new_input)
print(predictions)