#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


pip install mlxtend


# In[3]:


from mlxtend.frequent_patterns import apriori,association_rules


# In[4]:


from mlxtend.preprocessing import TransactionEncoder


# In[5]:


book = pd.read_csv("book.csv")


# In[6]:


book .head()


# In[7]:


df=pd.get_dummies(book)


# In[8]:


df.head()


# In[9]:


frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
frequent_itemsets


# In[10]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.7)
rules


# In[11]:


rules.sort_values('lift',ascending = False)[0:20]


# In[12]:


rules[rules.lift>1]


# In[ ]:




