#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import pandas as pd
import nltk
from bs4 import BeautifulSoup
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

import re


# # Import Dataset

# In[24]:


# Import Train Dataset

data = pd.read_csv("train.csv", header=0)

data = data[data['text'].notnull()]
data = data[data['target'].notnull()]

data.head(20)


# In[ ]:


# # Import Test Dataset

# data = pd.read_csv("test.csv", header=0)

# data = data[data['text'].notnull()]
# data = data[data['id'].notnull()]

# data.head(20)


# # Remove HTML

# In[25]:


def remove_html(text):
    soup = BeautifulSoup(text, 'html.parser')
    html_free = soup.get_text()
    return html_free


# In[26]:


data['text'] = data['text'].apply(lambda x:remove_html(x))


# # Remove http

# In[27]:


def remove_http(text):
    no_http = re.sub(r"http\S+", '', text)
    return no_http


# In[28]:


data['text'] = data['text'].apply(lambda x:remove_http(x))


# # Remove punctuation:

# In[29]:


def remove_punc(text):
    no_p = "". join([c for c in text if c not in string.punctuation])
    return no_p


# In[30]:


data['text'] = data['text'].apply(lambda x:remove_punc(x))


# # Remove Digits

# In[31]:


def remove_digit(text):
    re_digit = re.sub(r"\w*\d\w*", ' ', text)
    return re_digit


# In[32]:


data['text'] = data['text'].apply(lambda x:remove_digit(x))


# # Tokenize:

# In[33]:


# Instantiate Tokenizer
tokenizer = RegexpTokenizer(r'\w+')


# ### converting all letters to lower case

# In[34]:


data['text'] = data['text'].apply(lambda x:tokenizer.tokenize(x.lower()))


# # Remove words that length is less than 3

# In[35]:


def remove_less3word(text):
    words = [w for w in text if len(w)>=3]
    return words


# In[36]:


data['text'] = data['text'].apply(lambda x: remove_less3word(x))


# # Lemmatizing:

# In[37]:


# Instantiate lemmatizer
lemmatizer = WordNetLemmatizer()


# In[38]:


def word_lemmatizer(text):
    lem_text = [lemmatizer.lemmatize(i) for i in text]
    return lem_text


# In[39]:


data['text'] = data['text'].apply(lambda x: word_lemmatizer(x))


# # Stemming

# In[40]:


# Instantiate Stemmer
stemmer = PorterStemmer()


# In[41]:


def word_stemmer(text):
    stem_text = " ". join([stemmer.stem(i) for i in text])
    return stem_text


# In[42]:


data['text'] = data['text'].apply(lambda x: word_stemmer(x))


# # Correcting

# In[43]:


from textblob import TextBlob

data['text'] = data['text'].apply(lambda x: str(TextBlob(x).correct()))


# In[45]:


data.to_csv("Preprocess_Train.csv")


# In[44]:


# data.to_csv("Preprocess_Test.csv")


# In[ ]:





# In[ ]:




