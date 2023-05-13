#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/advertising.csv")
data = pd.read_csv("FutureSalesPrediction.csv")

#print(data.head())


# In[ ]:





# In[14]:


data= data.drop(['Unnamed: 0'],axis=1)


# In[15]:


data


# In[16]:


print(data.isnull().sum())


# In[17]:


import plotly.express as px
import plotly.graph_objects as go
figure = px.scatter(data_frame = data, x="Sales",y="TV", size="TV", trendline="ols")
figure.show()


# In[18]:


figure = px.scatter(data_frame = data, x="Sales", y="Newspaper", size="Newspaper", trendline="ols")
figure.show()


# In[19]:


figure = px.scatter(data_frame = data, x="Sales", y="Radio", size="Radio", trendline="ols")
figure.show()


# In[20]:


correlation = data.corr()
print(correlation["Sales"].sort_values(ascending=False))


# In[21]:


x = np.array(data.drop(["Sales"], 1))
y = np.array(data["Sales"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y,test_size=0.2, random_state=42)


# In[24]:


xtrain


# In[25]:


model = LinearRegression()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))


# In[26]:


#features = [[TV, Radio, Newspaper]]
features = np.array([[230.1, 37.8, 69.2]])
print(model.predict(features))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




