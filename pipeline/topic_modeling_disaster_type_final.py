# + tags=["parameters"]
# declare a list tasks whose products you want to use as inputs
upstream = ['vectorizer_countVec']


# + tags=["injected-parameters"]
# This cell was injected automatically based on your stated upstream dependencies (cell above) and pipeline.yaml preferences. It is temporary and will be removed when you save this notebook
random_seed = 42
upstream = {
    "vectorizer_countVec": {
        "nb": "C:\\Users\\gillrobe\\DataScience\\umads_697_data_medics\\pipeline\\output\\vectorizer_countVec.ipynb",
        "vectorizer": "C:\\Users\\gillrobe\\DataScience\\umads_697_data_medics\\pipeline\\output\\vectorizer_countVec.pkl",
    }
}
product = {
    "nb": "C:\\Users\\gillrobe\\DataScience\\umads_697_data_medics\\pipeline\\output\\topic_modeling_disaster_type_final.ipynb",
    "lda_model_earthquake": "C:\\Users\\gillrobe\\DataScience\\umads_697_data_medics\\pipeline\\output\\lda_model_earthquake.pkl",
    "lda_model_fire": "C:\\Users\\gillrobe\\DataScience\\umads_697_data_medics\\pipeline\\output\\lda_model_fire.pkl",
    "lda_model_flood": "C:\\Users\\gillrobe\\DataScience\\umads_697_data_medics\\pipeline\\output\\lda_model_flood.pkl",
    "lda_model_hurricane": "C:\\Users\\gillrobe\\DataScience\\umads_697_data_medics\\pipeline\\output\\lda_model_hurricane.pkl",
    "lda_topics_disaster_type": "C:\\Users\\gillrobe\\DataScience\\umads_697_data_medics\\pipeline\\output\\lda_topics_disastertype.csv",
}


# + [markdown] tags=[]
# importing all the functions defined in tools_rjg.py

# + tags=[]
from tools_rjg import *

# + tags=[]
import numpy as np
import pandas as pd
import re, nltk, spacy, gensim
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
nltk.download('punkt')
from gensim.corpora import Dictionary
from tqdm import tqdm
import os
from gensim.models import LdaModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import scikitplot as skplt
from sklearn.metrics import (precision_recall_curve,PrecisionRecallDisplay)
from sklearn.metrics import confusion_matrix
import pickle

# + tags=[]
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

# + tags=[]
import matplotlib.pyplot as plt
# %matplotlib inline

# + tags=[]
# # + tags=["parameters"]
upstream = []
random_seed = 42

# + tags=[]



# + tags=[]
df_all = generate_disaster_type_dataframe(disaster_types = ('earthquake', 'fire', 'flood', 'hurricane')
                                          , dev_train_test= ('dev', 'train', 'test'))

# + tags=[]



# + tags=[]
df_all['tweet_text_cleaned'] = df_all['tweet_text'].apply(lambda x: tweet_preprocessing(x))
df_all['lemmatized'] = df_all['tweet_text_cleaned'].apply(lambda x: lemmatize_tweet_text(x, allowed_postags=('NOUN', 'ADJ', 'VERB', 'ADV')))

# + tags=[]
df_all.sample(100)

# + [markdown] tags=[]
# ## LDA Model

# + tags=[]
params_earthquake = {'n_components' : 3,
          'max_iter' : 10,
          'learning_method' : 'online',
          'random_state' : random_seed,
          'batch_size' : 128,
          'evaluate_every' : -1,
          'n_jobs' : 1,
          'learning_decay' : .5}

params_fire = {'n_components' : 3,
          'max_iter' : 10,
          'learning_method' : 'online',
          'random_state' : random_seed,
          'batch_size' : 128,
          'evaluate_every' : -1,
          'n_jobs' : 1,
          'learning_decay' : .5}

params_flood = {'n_components' : 3,
          'max_iter' : 10,
          'learning_method' : 'online',
          'random_state' : random_seed,
          'batch_size' : 128,
          'evaluate_every' : -1,
          'n_jobs' : 1,
          'learning_decay' : .5}

params_hurricane = {'n_components' : 3,
          'max_iter' : 10,
          'learning_method' : 'online',
          'random_state' : random_seed,
          'batch_size' : 128,
          'evaluate_every' : -1,
          'n_jobs' : 1,
          'learning_decay' : .5}

lda_model_earthquake = LatentDirichletAllocation(**params_earthquake)
lda_model_fire = LatentDirichletAllocation(**params_fire)
lda_model_flood = LatentDirichletAllocation(**params_flood)
lda_model_hurricane = LatentDirichletAllocation(**params_hurricane)



# + tags=[]
with open("output\\vectorizer_countVec.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# + tags=[]
earthquake_text_vectorized = vectorizer.transform(list(df_all[df_all['disaster_type']=='earthquake']['lemmatized']))
fire_text_vectorized = vectorizer.transform(list(df_all[df_all['disaster_type']=='fire']['lemmatized']))
flood_text_vectorized = vectorizer.transform(list(df_all[df_all['disaster_type']=='flood']['lemmatized']))
hurricane_text_vectorized = vectorizer.transform(list(df_all[df_all['disaster_type']=='hurricane']['lemmatized']))

# + tags=[]
lda_model_earthquake.fit_transform(earthquake_text_vectorized)

# + tags=[]
lda_model_fire.fit_transform(fire_text_vectorized)

# + tags=[]
lda_model_flood.fit_transform(flood_text_vectorized)

# + tags=[]
lda_model_hurricane.fit_transform(hurricane_text_vectorized)

# + tags=[]
with open(product['lda_model_earthquake'], "wb") as f:
    pickle.dump(lda_model_earthquake, f)
with open(product['lda_model_fire'], "wb") as f:
    pickle.dump(lda_model_fire, f)
with open(product['lda_model_flood'], "wb") as f:
    pickle.dump(lda_model_flood, f)
with open(product['lda_model_hurricane'], "wb") as f:
    pickle.dump(lda_model_hurricane, f)

# + [markdown] tags=[]
# ## Performance Stats

# + tags=[]
# Log Likelyhood: Higher the better
print("Log Likelihood Earthquake: ", lda_model_earthquake.score(earthquake_text_vectorized))
print("Log Likelihood Fire: ", lda_model_fire.score(fire_text_vectorized))
print("Log Likelihood Flood: ", lda_model_flood.score(flood_text_vectorized))
print("Log Likelihood Hurricane: ", lda_model_hurricane.score(hurricane_text_vectorized))

# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
print("Perplexity Earthquake: ", lda_model_earthquake.perplexity(earthquake_text_vectorized))
print("Perplexity Fire: ", lda_model_fire.perplexity(fire_text_vectorized))
print("Perplexity Flood: ", lda_model_flood.perplexity(flood_text_vectorized))
print("Perplexity Hurricane: ", lda_model_hurricane.perplexity(hurricane_text_vectorized))

# + [markdown] tags=[]
# ## Extract Topic Keywords

# + tags=[]



# + tags=[]
topics_earthquake = pd.DataFrame(show_topics(fitted_vectorizer=vectorizer, fitted_lda_model=lda_model_earthquake, n_words=100, dname='earthquake'))
topics_fire = pd.DataFrame(show_topics(fitted_vectorizer=vectorizer, fitted_lda_model=lda_model_fire, n_words=100, dname='fire'))
topics_flood = pd.DataFrame(show_topics(fitted_vectorizer=vectorizer, fitted_lda_model=lda_model_flood, n_words=100, dname='flood'))
topics_hurricane = pd.DataFrame(show_topics(fitted_vectorizer=vectorizer, fitted_lda_model=lda_model_hurricane, n_words=100, dname='hurricane'))

df_topic_keywords = pd.concat([topics_earthquake,topics_fire,topics_flood,topics_hurricane],axis=0)
    # df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
    # df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
df_topic_keywords = df_topic_keywords.transpose()
df_topic_keywords.head()
# + tags=[]



# + tags=[]
df_topic_keywords.to_csv(product['lda_topics_disaster_type'])

# + tags=[]

