#!/usr/bin/env python
# coding: utf-8

# In[42]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')


# In[3]:


kc_house = pd.read_csv('kc_house_data.csv')
kc_house.head()


# #### Data Preprocessing :
# Data Cleaning : Looking for missing values and cleaning them if found
# 
# Data Tranformation : Transforming all the categorical object type values to numerical values.

# In[6]:


# Finding the Dataset columns
kc_house.columns


# In[8]:


# Finding the Dataset shape
kc_house.shape


# In[11]:


# checking for duplicates
kc_house.duplicated()


# In[12]:


# Checking the type of columns
kc_house.dtypes


# In[13]:


# checking for missing values
kc_house.isnull().sum()


# In[14]:


kc_house['sqft_above'].fillna(kc_house['sqft_above'].mean(), inplace=True)
kc_house.isnull().sum()


# ##### The only object feature here is date which will be dropped because we're not dealing with timeseries
# ##### After that all of our features are numerical so we can move forward

# In[16]:


kc_house.info()


# In[18]:


kc_house.drop(columns=['id','date'], axis=1,inplace=True)


# In[19]:


kc_house.info()


# In[20]:


# Splitting our dataset into input and output


# In[21]:


X = kc_house.drop(['price'], axis=1)
X.shape


# In[22]:


y = kc_house['price']
y.shape


# In[23]:


X.head()


# In[24]:


y.head()


# In[25]:


# Apply train_test_split


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


X_train,X_test,y_train,y_test = train_test_split(X,y,train_size= 0.8, random_state = 42)


# In[28]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# #### Model Building

# In[29]:


from sklearn.linear_model import LinearRegression


# In[30]:


# creating the object of the linear regression class
model1 = LinearRegression()


# In[31]:


# training phase of the model
model1.fit(X_train,y_train)


# In[32]:


# testing phase of the model
y_pred = model1.predict(X_test)


# In[33]:


y_pred


# In[34]:


y_test


# #### performance of the model - r2 score

# In[35]:


from sklearn.metrics import r2_score


# In[36]:


print(r2_score(y_test,y_pred))


# In[37]:


# my model is 70.07 % accurate with linear Regression


# #### intercept value is

# In[38]:


model1.intercept_


# In[44]:


pickle.dump(model1, open('model.pkl','wb'))


# In[45]:


model = pickle.load(open('model.pkl','rb'))


# In[51]:


print(model.predict(X))


# In[ ]:




