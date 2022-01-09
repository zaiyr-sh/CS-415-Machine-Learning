# How to Save a Seaborn Plot as a File

import pandas as pd

# Load irisAll.csv files as a Pandas DataFrame
# https://www.kaggle.com/arshid/iris-flower-dataset
irisData = pd.read_csv("hw_7/IRIS.csv")

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.pairplot(irisData, hue = "species", size = 3)

#plt.show()
plt.savefig('hw_7/lab_2/irisplot.png') # PNG, PDF, EPS, TIFF
plt.savefig('hw_7/lab_2/irisplot2-300dpi.pdf',dpi = 300)
irisData.plot(kind ="scatter",
    x ='sepal_length',
    y ='petal_length'
)
plt.grid()

#plt.show()
plt.savefig('hw_7/lab_2/irisplot3.png') # PNG, PDF, EPS, TIFF

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.imshow(mpimg.imread('hw_7/lab_2/irisplot.png'))
plt.show()