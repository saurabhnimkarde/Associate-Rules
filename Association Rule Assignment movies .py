#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


from mlxtend.frequent_patterns import apriori,association_rules


# In[3]:


from mlxtend.preprocessing import TransactionEncoder


# In[4]:


movie = pd.read_csv("my_movies.csv")


# In[5]:


movie .head()


# In[6]:


movie.shape


# In[7]:


df=pd.get_dummies(movie)


# In[8]:


df.head()


# In[10]:


movie.shape


# In[11]:


frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
frequent_itemsets


# In[12]:


rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
rules


# In[13]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.7)
rules


# In[14]:


rules.sort_values('lift',ascending = False).head(20)


# In[15]:


rules[rules.lift>1]


# In[ ]:




