### Experiment I: Using the DT model to classify the iris flowers

# Loading the training data and some exploration for the data
import pandas as pd
import matplotlib.pyplot as plt

# Load irisAll.csv files as a Pandas DataFrame
data = pd.read_csv("hw_5/IRIS.csv")

# Some information about dataset
print (data.shape)
print(type(data))
data.dtypes

data.dtypes
data.head()
data.describe()

# Prepare data for training models
labels = data.pop('species')
train=data

train.head()
labels.head()

### Training the Decision tree model and Displaying the trained model

# Training the decisin tree (DT) model
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
dt = DecisionTreeClassifier()
dt.fit(train, labels)
y_pred = dt.predict(train)
print(accuracy_score(labels, y_pred))

# Text Representation of the trained DT model
from sklearn import tree
text_representation = tree.export_text(dt)
print(text_representation)
# you can save it to the text file
with open("hw_5/decistion_tree.txt", "w") as fout:fout.write(text_representation)

# Graphical Representation of the trained DT model
fig = plt.figure(figsize=(16,4))
p = tree.plot_tree(dt,
    feature_names=train.columns, # train is pandas.core.frame.DataFrame
    class_names=labels.name, # labels is pandas.core.series.Series
    filled=True
)
plt.show()

# you can save the figure
fig.savefig("hw_5/decistion_tree.png")

### Scoring the trained model (Using the trained model for the prediction)

# Scoring the Trained DT model (Using the model for prediction)
new_input=[[3,4,5,4]] # predict function required 2D array input.
predictions = dt.predict(new_input)
print ('Input', new_input,'and Prediction', predictions)