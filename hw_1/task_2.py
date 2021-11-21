import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# load dataset as a panda data frame object
df = pd.read_csv('hw_1/Fish.csv')
df = df.loc[df['Species'] == 'Perch']

x=df[['Length1', 'Length2', 'Length3', 'Height']].values
y=df['Width'].values

# Implement the model
model = LinearRegression()

# Train (fit) the model
model.fit(x,y)

# Evaluate the model
r_sq = model.score(x,y)
print('Quality score of R squared value for the fitted model: ', round(r_sq, 2))

# Use the trained (fitted) model for prediction
xnew = np.array([[23, 24, 25, 15]])
y_pred = model.predict(xnew)
print('Estimated Width of that fish: ', round(y_pred[0], 3))