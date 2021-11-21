import pandas as pd
from sklearn.linear_model import LinearRegression

# load dataset as a panda data frame object
df = pd.read_csv('hw_1/Fish.csv')
df = df.loc[df['Species'] == 'Bream']

x=df['Height'].values
y=df['Width'].values

# Modify the input data shape by "reshaping" input data
x = x.reshape(-1, 1)

# Implement the model
model = LinearRegression()

# Train (fit) the model
model.fit(x,y)

# Evaluate the model
r_sq = model.score(x,y)
print('Quality score of R squared value for the fitted model: ', round(r_sq, 2))

# Use the trained (fitted) model for prediction
y_pred = model.predict([[5.55]])
print('Estimated Width of that fish: ', y_pred)