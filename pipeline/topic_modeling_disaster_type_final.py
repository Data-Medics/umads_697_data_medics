# importing all the functions defined in tools_rjg.py

from tools_rjg import *

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

from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

import matplotlib.pyplot as plt
# %matplotlib inline

random_seed = 10

#tags= ["parameters"]
upstream = []


# ## Extract and Aggregate all the disaster tweets by disaster type

df_all = generate_disaster_type_dataframe(disaster_types = ('earthquake', 'fire', 'flood', 'hurricane') 
                                          ,dev_train_test= ('dev', 'train', 'test'))

df_all.head()
df_all = df_all.sample(5000)



# ## Pre- Process Text

df_all['tweet_text_cleaned'] = df_all['tweet_text'].apply(lambda x: tweet_preprocessing(x))
df_all.head()






# ## Lemmatizer

# Do lemmatization keeping only Noun, Adj, Verb, Adverb
df_all['lemmatized'] = df_all['tweet_text_cleaned'].apply(lambda x: lemmatize_tweet_text(x, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']))
df_all.head()

src_file_path = os.path.dirname(os.path.abspath("__file__"))
filename_lemmatized = os.path.join(src_file_path, 'output\lemmatized_text.csv')
df_all.to_csv(filename_lemmatized)

# ## Vectorize

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words |= {"attach", "ahead", "rt",'www','tinyurl','com', 'https', 'http', 
               '&amp','amp', 'rt', 'bit', 'ly', 'bitly', 'trump', 'byte', 'bytes', 
               'donald','emoji','earthquake','fire','flood','hurricane'}
# +
# Build Disaster Type LDA Model
disaster_types = ['earthquake','fire', 'flood', 'hurricane']



vectorizer_params = {'analyzer' : 'word',
                    'min_df' : 10,
                    'stop_words' : stop_words,
                    'lowercase' : False,
                    'ngram_range' : (1,2)}

vectorizer = CountVectorizer(**vectorizer_params).fit(list(df_all['lemmatized']))
# -


# ## LDA Model

# +
params_earthquake = {'n_components' : 3,
          'max_iter' : 10,
          'learning_method' : 'online',
          'random_state' : random_seed,
          'batch_size' : 128,
          'evaluate_every' : -1,
          'n_jobs' : -1,
          'learning_decay' : .5}

params_fire = {'n_components' : 3,
          'max_iter' : 10,
          'learning_method' : 'online',
          'random_state' : random_seed,
          'batch_size' : 128,
          'evaluate_every' : -1,
          'n_jobs' : -1,
          'learning_decay' : .5}

params_flood = {'n_components' : 3,
          'max_iter' : 10,
          'learning_method' : 'online',
          'random_state' : random_seed,
          'batch_size' : 128,
          'evaluate_every' : -1,
          'n_jobs' : -1,
          'learning_decay' : .5}

params_hurricane = {'n_components' : 3,
          'max_iter' : 10,
          'learning_method' : 'online',
          'random_state' : random_seed,
          'batch_size' : 128,
          'evaluate_every' : -1,
          'n_jobs' : -1,
          'learning_decay' : .5}

lda_model_earthquake = LatentDirichletAllocation(**params_earthquake)
lda_model_fire = LatentDirichletAllocation(**params_fire)
lda_model_flood = LatentDirichletAllocation(**params_flood)
lda_model_hurricane = LatentDirichletAllocation(**params_hurricane)

# -


earthquake_text_vectorized = vectorizer.transform(list(df_all[df_all['disaster_type']=='earthquake']['lemmatized']))
fire_text_vectorized = vectorizer.transform(list(df_all[df_all['disaster_type']=='fire']['lemmatized']))
flood_text_vectorized = vectorizer.transform(list(df_all[df_all['disaster_type']=='flood']['lemmatized']))
hurricane_text_vectorized = vectorizer.transform(list(df_all[df_all['disaster_type']=='hurricane']['lemmatized']))

lda_model_earthquake.fit_transform(earthquake_text_vectorized)
lda_model_fire.fit_transform(fire_text_vectorized)
lda_model_flood.fit_transform(flood_text_vectorized)
lda_model_hurricane.fit_transform(hurricane_text_vectorized)

# ## Performance Stats

# +
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
# -

# ## Dominant Topics

# +
get_dominant_topics(lda_model_earthquake, earthquake_text_vectorized)

get_dominant_topics(lda_model_fire, fire_text_vectorized)

get_dominant_topics(lda_model_flood, flood_text_vectorized)

get_dominant_topics(lda_model_hurricane, hurricane_text_vectorized)
# -

# ## Topic Distribution



get_topic_distribution(lda_model_earthquake, vectorizer)

get_topic_distribution(lda_model_fire, vectorizer)

get_topic_distribution(lda_model_flood, vectorizer)

get_topic_distribution(lda_model_hurricane, vectorizer)

# ## Show top n keywords

show_topics(fitted_lda_model=lda_model_earthquake,fitted_vectorizer=vectorizer, n_words=20)

show_topics(fitted_lda_model=lda_model_fire,fitted_vectorizer=vectorizer, n_words=20)

show_topics(fitted_lda_model=lda_model_flood,fitted_vectorizer=vectorizer, n_words=20)

show_topics(fitted_lda_model=lda_model_hurricane,fitted_vectorizer=vectorizer, n_words=20)

# ## Extract Topic Keywords



# +
topics_earthquake = pd.DataFrame(show_topics(vectorizer=earthquake_vectorizer, lda_model=lda_model_earthquake, n_words=100, dname='earthquake'))
topics_fire = pd.DataFrame(show_topics(vectorizer=fire_vectorizer, lda_model=lda_model_fire, n_words=100, dname='fire'))
topics_flood = pd.DataFrame(show_topics(vectorizer=flood_vectorizer, lda_model=lda_model_flood, n_words=100, dname='flood'))
topics_hurricane = pd.DataFrame(show_topics(vectorizer=hurricane_vectorizer, lda_model=lda_model_hurricane, n_words=100, dname='hurricane'))

df_topic_keywords = pd.concat([topics_earthquake,topics_fire,topics_flood,topics_hurricane],axis=0)
    # df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
    # df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
df_topic_keywords = df_topic_keywords.transpose()
df_topic_keywords.head()
# -



topics_file = os.path.join(src_file_path, 'output\lda_topics_disastertype.csv')
df_topic_keywords.to_csv(topics_file)


