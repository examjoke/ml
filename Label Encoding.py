#!/usr/bin/env python
# coding: utf-8

# <h1 style="color:black;"> Label encoding</h1>
# 

# In[1]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder


# In[2]:


cancer_data = pd.read_csv('breast-cancer.csv')
print(cancer_data.sample(5))


# In[3]:


print(cancer_data['diagnosis'].value_counts())


# In[4]:


label_encode = LabelEncoder()

labels = label_encode.fit_transform(cancer_data['diagnosis'])
cancer_data['target'] = labels
print(cancer_data.sample(5))


# In[5]:


print(cancer_data['target'].value_counts())


# <h1 style="color:black;">Iris dataset label encoding</h1>

# In[6]:


iris_data = pd.read_csv('iris.csv')


print(iris_data.head())


# In[7]:


print(iris_data['Species'].value_counts())


# In[8]:


label_encode = LabelEncoder()

iris_labels = label_encode.fit_transform(iris_data['Species'])
iris_data['target'] = iris_labels


# In[9]:


print(iris_data['target'].value_counts())

