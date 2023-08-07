#!/usr/bin/env python
# coding: utf-8

# In[26]:


import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as met
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso


# In[27]:


df = pd.read_csv(r'diamonds.csv',index_col=0)
df


# In[28]:


df.dtypes


# In[29]:


df['cut'].value_counts()


# In[30]:


df['cut'] = df['cut'].map({'Ideal': 1, 'Premium': 2, 'Very Good': 3, 'Good': 4, 'Fair': 5})


# In[31]:


df['cut']


# In[32]:


df['color'].value_counts()


# In[33]:


df['color'] = df['color'].map({'D': 1, 'E': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6, 'J': 7})


# In[34]:


df['color'].value_counts()


# In[35]:


df['clarity'].value_counts()


# In[36]:


df['clarity'] = df['clarity'].map({'IF': 1, 'VVS1': 2, 'VVS2': 3, 'VS1': 4, 'VS2': 5, 'SI1': 6, 'SI2': 7 , 'I1' : 8})


# In[37]:


df.describe


# In[38]:


y = df ['price'] 


# In[39]:


y


# In[40]:


x = df.loc[:,(df.columns != 'price')]
x


# In[41]:


xtrain,xtest,ytrain,ytest =train_test_split(x, y, test_size = 0.30, random_state = 42)


# In[42]:


model = LinearRegression()
df.describe()


# In[43]:


model.fit(xtrain, ytrain)


# In[44]:


ypred = model.predict(xtest)


# In[45]:


mean_absolute_error(ytest, ypred)


# In[46]:


met.mean_squared_error(ytest, ypred)


# In[47]:


plt.scatter(ytest,ypred)


plt.xlabel('prices')
plt.ylabel('predicted prices')
plt.title('Price vs Predicted Price')


# In[63]:


pipe = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('lasso', Lasso(alpha=0.1))
])


# In[64]:


pipe.fit(x , y)


# In[65]:


c = pipe.named_steps['lasso'].coef_


# In[74]:


plt.plot(range(9),c)
plt.xticks(range(9),x)
plt.ylabel('coefficients')
plt.xlabel('features');


# In[ ]:




