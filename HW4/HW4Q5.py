import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# features
Outlook = [0,0,1,1,2,2] # Outlook Catelogies-- Sunny:0, Overcast:1, Rainy:2
Temperature = [1,1,1,0,0,0] # Temp Catelogies-- High:1, Low:0
Humidity = [1,1,0,0,1,1] # Humidity Catelogies-- High:1, Low:0
Windy = [0,1,0,1,0,1] # Windy Catelogies-- Yes:1, No:0
# target
Decision = [0,0,1,1,1,0] # Decision Catelogies-- Play:1, No Play:0
# Build Dataset
data = {"Outlook":Outlook,"Temperature":Temperature,"Humidity":Humidity,"Windy":Windy,"Decision":Decision}
df = pd.DataFrame(data, index=[1,2,3,4,5,6])

from sklearn import tree
# split the dataset into features and targets
features = df.iloc[:,0:4].to_numpy()
targets = df.iloc[:,4].to_numpy()

# do CART, a type of decision tree classification
cart = tree.DecisionTreeClassifier(criterion="gini")
cart.fit(features, targets)

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=200)
# plot a decision tree
tree.plot_tree(cart, feature_names=["outlook","temp","humd","windy"],
               class_names=["play","noplay"],filled=True,rounded=True)
fig.savefig('image.png')