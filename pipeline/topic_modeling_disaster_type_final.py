# + tags=["parameters"]
# # + tags=["parameters"]
# declare a list tasks whose products you want to use as inputs
upstream = ['vectorizer_countVec']
random_seed = 42
# -






# importing all the functions defined in tools_rjg.py

from tools_rjg import *

import pandas as pd
import re, nltk, spacy, gensim
nltk.download('punkt')
import pickle

from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD

import matplotlib.pyplot as plt
# %matplotlib inline







df_all = generate_disaster_type_dataframe(disaster_types = ('earthquake', 'fire', 'flood', 'hurricane')
                                          , dev_train_test= ('dev', 'train', 'test'))




df_all['tweet_text_cleaned'] = df_all['tweet_text'].apply(lambda x: tweet_preprocessing(x))
df_all['lemmatized'] = df_all['tweet_text_cleaned'].apply(lambda x: lemmatize_tweet_text(x, allowed_postags=('NOUN', 'ADJ', 'VERB', 'ADV')))

df_all.sample(100)

# ## LDA Model

# +
params_earthquake = {'n_components' : 2,
          'max_iter' : 10,
          'learning_method' : 'online',
          'random_state' : random_seed,
          'batch_size' : 128,
          'evaluate_every' : -1,
          'n_jobs' : 1,
          'learning_decay' : .5}

params_fire = {'n_components' : 4,
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

params_hurricane = {'n_components' : 5,
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

# -


with open("output/vectorizer_countVec.pkl", "rb") as f:
    vectorizer = pickle.load(f)

earthquake_text_vectorized = vectorizer.transform(list(df_all[df_all['disaster_type']=='earthquake']['lemmatized']))
fire_text_vectorized = vectorizer.transform(list(df_all[df_all['disaster_type']=='fire']['lemmatized']))
flood_text_vectorized = vectorizer.transform(list(df_all[df_all['disaster_type']=='flood']['lemmatized']))
hurricane_text_vectorized = vectorizer.transform(list(df_all[df_all['disaster_type']=='hurricane']['lemmatized']))

lda_model_earthquake.fit_transform(earthquake_text_vectorized)

lda_model_fire.fit_transform(fire_text_vectorized)

lda_model_flood.fit_transform(flood_text_vectorized)

lda_model_hurricane.fit_transform(hurricane_text_vectorized)

with open(product['lda_model_earthquake'], "wb") as f:
    pickle.dump(lda_model_earthquake, f)
with open(product['lda_model_fire'], "wb") as f:
    pickle.dump(lda_model_fire, f)
with open(product['lda_model_flood'], "wb") as f:
    pickle.dump(lda_model_flood, f)
with open(product['lda_model_hurricane'], "wb") as f:
    pickle.dump(lda_model_hurricane, f)

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

# ## Extract Topic Keywords

# +
topics_earthquake = pd.DataFrame(show_topics(fitted_vectorizer=vectorizer, fitted_lda_model=lda_model_earthquake, n_words=100, dname='earthquake'))
topics_fire = pd.DataFrame(show_topics(fitted_vectorizer=vectorizer, fitted_lda_model=lda_model_fire, n_words=100, dname='fire'))
topics_flood = pd.DataFrame(show_topics(fitted_vectorizer=vectorizer, fitted_lda_model=lda_model_flood, n_words=100, dname='flood'))
topics_hurricane = pd.DataFrame(show_topics(fitted_vectorizer=vectorizer, fitted_lda_model=lda_model_hurricane, n_words=100, dname='hurricane'))

df_topic_keywords = pd.concat([topics_earthquake,topics_fire,topics_flood,topics_hurricane],axis=0)
    # df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
    # df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
df_topic_keywords = df_topic_keywords.transpose()
df_topic_keywords.head()
# -



df_topic_keywords.to_csv(product['lda_topics_disaster_type'])


