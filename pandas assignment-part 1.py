#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np
from datetime import datetime


# In[3]:


date=pd.date_range(start='1/1/2019', end='31/12/2019',freq='D')


# In[4]:


date


# In[5]:


date_df=pd.DataFrame(date,columns=['dates'])
date_df['random']=np.random.randint(0,100, size=len(date))


# In[6]:


date_df.head()


# 2. Given Pandas series , height = [23,42,55] and weight = [71,32,48] . Create a dataframe with height and weight as column names. 

# In[7]:


height = [23,42,55]
weight = [71,32,48]


# In[8]:


df=pd.DataFrame()


# In[9]:


df['height']=height


# In[10]:


df['weight']=weight


# In[11]:


df


# 3. How to get the items of series A not present in series B .From ser1 remove items present in ser2.

# In[12]:


ser1=pd.Series([2,3,4,5])
ser2=pd.Series([5,6,7,8])


# In[13]:


res=filter(lambda x: x not in ser2, ser1)
res


# In[14]:


import seaborn as sns


# In[15]:


titanic=sns.load_dataset('titanic')


# In[16]:


titanic.head()


# In[17]:


print(titanic.shape)


# In[18]:


titanic['age'].describe()


# In[19]:


titanic['sibsp'].value_counts()


# In[20]:


titanic['fare'].head()


# In[21]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[22]:


plt.scatter(titanic['fare'],titanic['class'])


# In[23]:


fare_grouped=titanic.groupby('class')


# In[24]:


titanic_first=fare_grouped.get_group('First')

titanic_second=fare_grouped.get_group('Second')
titanic_third=fare_grouped.get_group('Third')


# In[25]:


titanic_second[:10]


# In[26]:


sns.boxplot(titanic_first['fare'])


# In[27]:


sns.boxplot(titanic_second['fare'])


# In[28]:


sns.boxplot(titanic_third['fare'])


# In[29]:


titanic['fare'].describe()


# In[30]:


(titanic['fare'].values>510).sum()


# In[31]:


titanic=titanic[titanic['fare']<=500]


# In[32]:


(titanic['fare'].values>500).sum()


# In[33]:


titanic['bins']=pd.cut(titanic.fare,[0,100,200,300,400,500])


# In[34]:


titanic


# ## 8.Count the number of missing values in each column?

# In[35]:


titanic.isnull().sum()


# ## 9.Get the row number of the 5th largest value in the Age column of titanic dataset?

# In[61]:


df_age_sorted=titanic.sort_values(['age'],ascending=False)


# In[62]:


df_age_sorted['age'].head()


# In[ ]:





# In[ ]:





# ## 10.normalize all columns in the Data frame,

# In[36]:


from sklearn import preprocessing


# In[54]:


data = [[1,2013,'1100001',157500],
       [2,2014,'1100002',458226],
       [3,2015,'1100003',458226],
       [4,2010,'1100004',456312],
       [5,2018,'1100005',235485]]
df=pd.DataFrame(data,columns=['Sr_No','Year','ID','Sales'])


# In[55]:


df.head()


# In[58]:


x=df.values
normalize=preprocessing.MinMaxScaler()
x_scaled=normalize.fit_transform(x)
df_normal=pd.DataFrame(x_scaled,columns=['Sr_No','Year','ID','Sales'])


# In[59]:


df_normal


# ### 12. How to convert a series of date-strings to a timeseries?

# In[ ]:





# In[ ]:




