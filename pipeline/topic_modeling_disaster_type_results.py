# + tags=["parameters"]
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
# -
# # + tags=["parameters"]
# declare a list tasks whose products you want to use as inputs
upstream = ['vectorizer_countVec','topic_modeling_disaster_type_final']
random_seed = 42

# + tags=[]
df_all = generate_disaster_type_dataframe(disaster_types = ('earthquake', 'fire', 'flood', 'hurricane')
                                          , dev_train_test= ('dev', 'train', 'test'))
# -

with open("output\\vectorizer_countVec.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# + tags=[]
with open("output\\lda_model_earthquake.pkl", "rb") as f:
    lda_model_earthquake = pickle.load(f)
with open("output\\lda_model_fire.pkl", "rb") as f:
    lda_model_fire = pickle.load(f)
with open("output\\lda_model_flood.pkl", "rb") as f:
    lda_model_flood = pickle.load(f)
with open("output\\lda_model_hurricane.pkl", "rb") as f:
    lda_model_hurricane = pickle.load(f)


# + tags=[]
df_all['tweet_text_cleaned'] = df_all['tweet_text'].apply(lambda x: tweet_preprocessing(x))
df_all['lemmatized'] = df_all['tweet_text_cleaned'].apply(lambda x: lemmatize_tweet_text(x, allowed_postags=('NOUN', 'ADJ', 'VERB', 'ADV')))

# + tags=[]
df_all.sample(100)

# + tags=[]
earthquake_text_vectorized = vectorizer.transform(list(df_all[df_all['disaster_type']=='earthquake']['lemmatized']))
fire_text_vectorized = vectorizer.transform(list(df_all[df_all['disaster_type']=='fire']['lemmatized']))
flood_text_vectorized = vectorizer.transform(list(df_all[df_all['disaster_type']=='flood']['lemmatized']))
hurricane_text_vectorized = vectorizer.transform(list(df_all[df_all['disaster_type']=='hurricane']['lemmatized']))

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
# -

get_dominant_topics(lda_model_earthquake,earthquake_text_vectorized)

get_dominant_topics(lda_model_fire,fire_text_vectorized)

get_dominant_topics(lda_model_flood,flood_text_vectorized)

get_dominant_topics(lda_model_hurricane, hurricane_text_vectorized)

# + [markdown] tags=[]
# ## Get Topic Distribution

# + tags=[]
get_topic_distribution(lda_model_earthquake, vectorizer)


# + tags=[]
get_topic_distribution(lda_model_fire, vectorizer)
# + tags=[]
get_topic_distribution(lda_model_flood, vectorizer)


# + tags=[]
get_topic_distribution(lda_model_hurricane, vectorizer)

# + [markdown] tags=[]
# ## Show Top N Topics
# -

show_topics(vectorizer, fitted_lda_model=lda_model_earthquake, n_words=20)

show_topics(vectorizer, fitted_lda_model=lda_model_fire, n_words=20)

show_topics(vectorizer, fitted_lda_model=lda_model_flood, n_words=20)

show_topics(vectorizer, fitted_lda_model=lda_model_hurricane, n_words=20)

# ## Extract most representative sentence for each topic

pd.options.display.max_colwidth = 1000

earthquake_top_sentence = extract_most_representative_sentence(lda_model_earthquake, earthquake_text_vectorized, df_all[df_all['disaster_type']=='earthquake'])
for topic in range(lda_model_earthquake.n_components):
    print('----------------------')
    print(f'Topic_{topic}', earthquake_top_sentence[f'Topic_{topic}'][1]['tweet_text'])

fire_top_sentence = extract_most_representative_sentence(lda_model_fire, fire_text_vectorized, df_all[df_all['disaster_type']=='fire'])
for topic in range(lda_model_fire.n_components):
    print('----------------------')
    print(f'Topic_{topic}', fire_top_sentence[f'Topic_{topic}'][1]['tweet_text'])



flood_top_sentence = extract_most_representative_sentence(lda_model_flood, flood_text_vectorized, df_all[df_all['disaster_type']=='flood'])
for topic in range(lda_model_flood.n_components):
    print('----------------------')
    print(f'Topic_{topic}', flood_top_sentence[f'Topic_{topic}'][1]['tweet_text'])



hurricane_top_sentence = extract_most_representative_sentence(lda_model_hurricane, hurricane_text_vectorized, df_all[df_all['disaster_type']=='hurricane'])
for topic in range(lda_model_hurricane.n_components):
    print('----------------------')
    print(f'Topic_{topic}', hurricane_top_sentence[f'Topic_{topic}'][1]['tweet_text'])



pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(lda_model_earthquake, earthquake_text_vectorized, vectorizer, mds='tsne')
panel


