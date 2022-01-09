from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

# load dataset as an panda data frame object
df = pd.read_csv('hw_9/diabetes.csv')
df.head()
df.dtypes
len(df.columns)
df.describe().T

X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age' ]] # 'Outcome'
#x = df[features]
print(type(X)) # pandas.core.frame.DataFrame
print(X)
print(X.shape)

y=df['Outcome'].values
print(type(y)) # numpy.ndarray
print(y)
print(y.shape)

# split the data using Scikit-Learn's train_test_split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 1)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

# We can inspect the computed means and standard deviations.
scaler.mean_
scaler.scale_

# scikit-learn convention: if an attribute is learned from the data, its name ends with an underscore
# (i.e. _), as in mean_ and scale_ for the StandardScaler.
data_Xtrain_scaled = scaler.transform(X_train)
data_Xtrain_scaled

data_Xtest_scaled = scaler.transform(X_test)
data_Xtest_scaled

# Notice that the mean of all the columns is close to 0 and
# the standard deviation in all cases is close to 1.
data_Xtrain_scaled.describe()
data_Xtest_scaled.describe()