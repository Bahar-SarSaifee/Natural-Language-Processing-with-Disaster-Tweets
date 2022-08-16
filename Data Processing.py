#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

import torch
from torch.utils.data import Dataset

from transformers import DistilBertTokenizerFast, BertTokenizer
from transformers import Trainer, TrainingArguments

import nltk, random
from nltk.corpus import movie_reviews

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

from lime.lime_text import LimeTextExplainer
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from keras.layers import *
from keras.models import Sequential, Model
import pydot
import pydotplus
from pydotplus import graphviz
from keras.utils.vis_utils import plot_model
from keras.utils.vis_utils import model_to_dot


# In[2]:


try:
    get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
    pass
import tensorflow as tf

import tensorflow_hub as hub

from tensorflow.keras import layers
import bert
import math

import os

seed_value = 42
os.environ['PYTHONHASHSEED']= str(seed_value)

import random as rn

np.random.seed(seed_value)
rn.seed(seed_value)


# # Creating a BERT Tokenizer

# In[3]:


BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/4",
                            trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)


# In[4]:


###################Importing Train Dataset##########################

train = pd.read_csv("Preprocess_Train.csv", header=0)
train = train[train['text'].notnull()]
train = train[train['target'].notnull()]
train.head()


# In[5]:


train.shape


# In[6]:


###################Split Train Dataset to Xtrain & Ytrain##########################

Xtrain = train['text']
Ytrain = train['target']


# In[7]:


###################Tokenizing & Padding of Train Dataset##########################

maxlen = 100

Xtrain = '[CLS]' + Xtrain + '[SEP]'
Xtrain = list(map(tokenizer.tokenize, Xtrain))

c = Xtrain

Xtrain = [tokenizer.convert_tokens_to_ids(txt) for txt in Xtrain]
Xtrain = pad_sequences(Xtrain, padding='post', maxlen=maxlen)
Xtrain = np.array(Xtrain)


# In[8]:


###################Importing Test Dataset##########################

test = pd.read_csv("Preprocess_Test.csv", header=0)
test = test[test['text'].notnull()]
test


# In[9]:


Xtest = test['text']
Ytest = test['target']


# In[10]:


###################Tokenizing & Padding of Test Dataset##########################

Xtest = '[CLS]' + Xtest + '[SEP]'
Xtest = list(map(tokenizer.tokenize, Xtest))
Xtest = [tokenizer.convert_tokens_to_ids(txt) for txt in Xtest]
Xtest = pad_sequences(Xtest, padding='post', maxlen=maxlen)
Xtest = np.array(Xtest)


# # Glove Embedding

# In[11]:


###################Loading Pretrained Word Embeddings to Create Feature Matrix##########################

from numpy import array
from numpy import asarray
from numpy import zeros

vocab_size = len(tokenizer.vocab) + 1

embeddings_dictionary = dict()
glove_file = open('data_embedding/glove/glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.vocab.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector


# In[12]:


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# In[13]:


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, embedding_matrix):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, weights=[embedding_matrix], output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


# # Creating the Model Using Deep Learning Techniques for Text Classification

# In[22]:


pooled_outputs1 = []
pooled_outputs2 = []
pooled_outputs3 = []

num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

embedding_dim = 100
embed_input = Input(shape=(maxlen,))

dropout=0.1
filter_sizes = [4,5,6]
num_filters = [80,100,200]

###################Embedding Layer##########################

# x = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=maxlen)(embed_input)
x = TokenAndPositionEmbedding(maxlen, vocab_size, embedding_dim, embedding_matrix)(embed_input)
x = TransformerBlock(embedding_dim, num_heads, ff_dim)(x)    
###################Attention Layer##########################

# for i in range(1):
#     att_output = MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(x, x)
#     att_output = Dropout(dropout)(att_output, training=False)
#     layer1_output = LayerNormalization(epsilon=1e-6)(att_output + x)
#     ff1_output = Sequential([Dense(ff_dim, activation="relu"),Dense(embedding_dim)])(layer1_output)
#     ff1_output = Dropout(dropout)(ff1_output, training=False)
#     t = LayerNormalization(epsilon=1e-6)(ff1_output + layer1_output)
    
#     pooled_outputs2.append(t)

# merge2 = concatenate(pooled_outputs2)

###################CNN Layers##########################
for i in range(1):
    conv = (Convolution1D(filters=num_filters[i],
                          kernel_size=filter_sizes[i],
                          padding="same",
                          activation="relu"))(x)
    pooled_outputs1.append(conv)

merge1 = concatenate(pooled_outputs1)

###################LSTM Layers##########################

x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(merge1)

x = Flatten()(merge1)
x = Dense(units=1, activation='sigmoid')(x)

model = Model(inputs=[embed_input] , outputs=[x])

# model.summary()


# In[23]:


# import pydot
# import pydotplus
# import tensorflow as tf
# from pydotplus import graphviz
# from keras.utils.vis_utils import plot_model
# from keras.utils.vis_utils import model_to_dot

# tf.keras.utils.plot_model(
#     model,
#     to_file='model.png',
#     show_shapes=False,
#     show_layer_names=True,
#     rankdir='TB',
#     expand_nested=False,
#     dpi=80
# )


# In[24]:


###################Compiling our Model##########################

opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])


# In[25]:


###################Training our Model##########################

plrty = ['0', '1']
result = []
preds = []

for i in range(20):
    print(i, ':')
    history = model.fit(Xtrain, Ytrain, validation_split=0.2, epochs=1, batch_size=20)
    preds.append(np.round(model.predict(Xtest)))
#     print(classification_report(Ytest, preds, target_names=plrty))
    results = f1_score(Ytest, preds[i], average='macro')
    print(i, 'result :',results)
    result.append(results)
    
print(result)


# In[ ]:


import numpy as np
# np.savetxt("submission-0.776.csv", preds[9], delimiter=",", fmt="%d")


# In[ ]:




