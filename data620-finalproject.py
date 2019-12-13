#!/usr/bin/env python
# coding: utf-8

# # Data 620 - Final Project
# 
# Baron Curtin, Heather Geiger
# 
# RNA sequencing has been a valuable tool in genetics research for well over a decade now. By looking at RNA, one can look at which genes are expressed in addition to just looking at genetic variation.
# 
# More recently, RNA sequencing at the single cell level rather than just of the whole tissue lets you see which genes are expressed in subsets of a tissue or particular cell types. Popular applications include development, neurology, autoimmune disease, and oncology.
# 
# In our final project, we plan to use automated data analysis to get a broad overview of the published research on single-cell RNA sequencing. Specifically, we plan to focus on two aspects of this. 
# 
# The first aspect will be to get a sense of the most influential authors in this field, and the relationships between them. To do this, we will run a network analysis, where connections are defined by authors being on the same paper together. 
# 
# The second aspect will be to get a sense of recurrent themes in the literature through text mining. Here, we will focus on recurrent terms in the abstracts to do this. Since the abstract of a paper is meant to provide a summary of the most important themes in a single paragraph or two, text mining these should provide a lot of information despite only needing to process a relatively small amount of text.

# In[1]:


# standard imports
import pickle
import gzip
from collections import Counter
from pathlib import Path

# third party imports
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer

from Bio import Entrez
from Bio import Medline
from tqdm import tqdm

# stop words
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# additional jupyter setup
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load Data
# 
# Lets load our data and perform some EDA to get a feel for our data.

# In[2]:


# load data
data_paths = (Path.cwd() / 'data').glob('*half*.gz')

def parse_data(data_path):
    with gzip.open(data_path, 'rb') as pf:
        loaded_data = pickle.load(pf)
    return loaded_data

data = [x for dp in data_paths for x in parse_data(dp)]
df = pd.DataFrame(data)


# ## Basic Analysis
# 
# We can perform some basic exploratory analysis to better understand the data we are working with

# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.head()


# ## Feature Extraction
# From the basic exploration, we are able to determine that there are various features we can create and features that we do
# not need for our analysis.
# 
# Lets create some features from our data to make analysis a bit easier and go into our secondary EDA. Some features we might
# look to create are separating the year from the data column, possibly separating each time interval from the date column, and
# isolating our authors so that we have one author per line

# In[6]:


# remove columns not needed for analysis
df = df[['PMID', 'FAU', 'AB', 'EDAT', 'TA', 'OT']]

# rename columns for explicitness
df.columns = ['KEY', 'AUTHORS', 'ABSTRACT', 'DATE', 'TITLE', 'KEYWORDS']
df.head()


# In[7]:


# create separate date columns
df['DATE_'] = pd.to_datetime(df['DATE'], infer_datetime_format=True)

# create year, month, day columns
df['YEAR'] = df['DATE_'].dt.year
df['MONTH'] = df['DATE_'].dt.month
df['DAY'] = df['DATE_'].dt.day
df['TIME'] = df['DATE_'].dt.time


# In[8]:


# create new dataframe based on authors
x_df = pd.DataFrame(df.AUTHORS.fillna('??????').str.join('|').str.split('|').tolist(), index=df.KEY).stack()
x_df = x_df.reset_index([0, 'KEY'])
x_df.columns = ['KEY', 'AUTHOR']

# join original data back to the new df
x_df = x_df.merge(df.drop(columns=['AUTHORS']), on=['KEY'])
x_df.head()


# In[9]:


# join keywords in a single list
x_df['KEYWORDS_MERGED'] = x_df['KEYWORDS'].str.join('|')


# In[17]:


## Data Analysis/Visualization
plt.figure(figsize=(20, 20), dpi=600)

# top 10 authors
plt.subplot(2, 2, 1)
top10authors = pd.DataFrame.from_records(Counter(x_df[x_df['AUTHOR'] != '?']['AUTHOR']).most_common(10), columns=['NAME', 'COUNT'])
sns.barplot(x='COUNT', y='NAME', data=top10authors)
plt.title('Top 10 Authors')

# publications over time
plt.subplot(2, 2, 2)
yearly = pd.DataFrame(df['YEAR'].value_counts().reset_index())
yearly.columns = ['YEAR', 'COUNT']
sns.lineplot(x='YEAR', y='COUNT', data=yearly)
plt.title('Publications Over Time')
plt.xlim([df.YEAR.min(), df.YEAR.max()])

# top 10 journals
plt.subplot(2, 2, 3)
top10journals = pd.DataFrame.from_records(Counter(df['TITLE']).most_common(10), columns=['TITLE', 'COUNT'])
sns.barplot(x='COUNT', y='TITLE', data=top10journals)
plt.title('Top 10 Journals')

# top 10 keywords
plt.subplot(2, 2, 4)
keywords = df['KEYWORDS'].dropna().str.join('|').str.split('|').to_list()
flat_kws = [_.lower() for kw in keywords
            for _ in kw]
top10kw = pd.DataFrame.from_records(Counter(flat_kws).most_common(10), columns=['KEYWORD', 'COUNT'])
sns.barplot(x='COUNT', y='KEYWORD', data=top10kw)
plt.title('Top 10 Keywords')


# Various trends are evident from the plots above, however it may be useful to create an additional dataframe index by the title,
# and then create plots of the keywords conditioned on the title. It may also be helpful to create plots for single words in
# an abstract, as well as 2-3 word phrases. Lets first create the dataframe which will allow us to do so, and then plot the results.

# In[ ]:




