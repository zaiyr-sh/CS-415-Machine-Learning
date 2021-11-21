import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# https://www.kaggle.com/ncchetan/market-basket-analysis-store-data
data = pd.read_csv('hw_4/store_data.csv', header = None)
data.head()

# Data Preprocessing
# The Apriori library we are going to use requires our dataset to be in the form of a list of lists
# rather than a data frame
records = []
for i in range(0, len(data)):
    records.append([str(data.values[i,j]) for j in range(0, 20)])

# Mining Association Rules using apriori algorithm
from apyori import apriori
# Let's suppose that we want rules for only those items that are purchased at least an average
# of 5 times a day, or 7 x 5 = 35 times in one week, since our dataset is for a one-week time
# period. The support for those items can be calculated as 35/7500 = 0.0045. The minimum
# confidence for the rules is 20% or 0.2. Similarly, we specify the value for lift as 3 and
# finally min_length is 2 since we want at least two products in our rules. These values are
# mostly just arbitrarily chosen, so you can play with these values and see what difference it
# makes in the rules you get back out.
association_rules = apriori(records, min_support = 0.0045, min_confidence = 0.2, min_lift = 3, min_length = 2)

# min_length: The minimum number of items in a rule
# max_length: The maximum number of items in a rule

# Converting the associations to lists
association_results = list(association_rules)

# Display the first rule in the association_rules list
print(association_results[0])

# Viewing the Results
# Display the total number of rules mined by the apriori algorithm
print(len(association_results))

#convert results in a dataframe format
df_results = pd.DataFrame(association_results)

# display the all rules
df_results.head()

# Method 1: list each rules with their support, confidence, and lift:
# inspired from https://stackabuse.com/association-rulemining-via-apriori-algorithm-in-python/
for item in association_results:
    # first index of the inner list
    # Contains base item and add item
    pair = item[0]
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")
print("------------------------------------------------------------------------------------")

# Method 2: list each rules with their support, confidence, and lift:
# inspired from https://github.com/pirandello/apriori/blob/master/Apriori.ipynb
results = []
for item in association_results:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0]
    items = [x for x in pair]

    value0 = str(items[0])
    value1 = str(items[1])

    #second index of the inner list
    value2 = str(item[1])[:7]

    #third index of the list located at 0th
    #of the third index of the inner list

    value3 = str(item[2][0][2])[:7]
    value4 = str(item[2][0][3])[:7]

    rows = (value0, value1, value2, value3, value4)
    results.append(rows)

labels = ['Item 1','Item 2','Support','Confidence','Lift']
product_suggestion = pd.DataFrame.from_records(results, columns = labels)
print(product_suggestion)   
print("------------------------------------------------------------------------------------")

# which rules did score the highest Lift?
print(product_suggestion[product_suggestion.Lift == product_suggestion.Lift.max()])
print("------------------------------------------------------------------------------------")
# which rules did score the highest Confidence?
print(product_suggestion[product_suggestion.Confidence == product_suggestion.Confidence.max()])
print("------------------------------------------------------------------------------------")
# which rules did score the highest Support?
print(product_suggestion[product_suggestion.Support == product_suggestion.Support.max()])