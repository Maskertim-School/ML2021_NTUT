#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd

# read the iris dataset which is csv format
col_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
iris = pd.read_csv("iris.data", header=None, names=col_names)
iris.head()


# In[5]:


# map iris class name to number
iris_class = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
iris['species_tag'] = [iris_class[i] for i in iris.species]
iris.tail()


# In[6]:


#split data into attributes and target/label
iris_attrs = iris.drop(['species','species_tag'], axis=1)
iris_labels = iris.species_tag


# In[7]:


print(iris_attrs)


# In[17]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# avg of score
avg = 0

# run 10 times
for i in range(10):
    # split data into training and testing sets
    train_data, test_data, train_label, test_label = train_test_split(iris_attrs, iris_labels,
                                                                      random_state=None, train_size=0.7)
    # set 5 neighbors of knn
    knn_5 = KNeighborsClassifier(n_neighbors = 5)
    # fit the model on the training data
    knn_5.fit(train_data, train_label)
    # see how the model preforms
    avg = avg + knn_5.score(test_data, test_label)

# average accuracy
print(avg/10)


# In[19]:


# predicted label and actual label
print('predict:',knn_5.predict(test_data)[0:10],'actual:',test_label.tolist()[0:10])

