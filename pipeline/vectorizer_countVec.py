# + tags=[]
from tools_rjg import *
import numpy as np
import pandas as pd
import re, nltk, spacy, gensim
from nltk.corpus import stopwords
nltk.download('punkt')
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle
import os


# + tags=["parameters"]
upstream = []
random_seed = 42

# + tags=["injected-parameters"]
# This cell was injected automatically based on your stated upstream dependencies (cell above) and pipeline.yaml preferences. It is temporary and will be removed when you save this notebook
random_seed = 42
product = {
    "nb": "C:\\Users\\gillrobe\\DataScience\\umads_697_data_medics\\pipeline\\output\\vectorizer_countVec.ipynb",
    "vectorizer": "C:\\Users\\gillrobe\\DataScience\\umads_697_data_medics\\pipeline\\output\\vectorizer_countVec.pkl",
}


# + tags=[]
df_all = generate_disaster_type_dataframe(disaster_types = ('earthquake', 'fire', 'flood', 'hurricane')
                                          , dev_train_test= ('dev', 'train', 'test'))

# + tags=[]



# + tags=[]
df_all['tweet_text_cleaned'] = df_all['tweet_text'].apply(lambda x: tweet_preprocessing(x))
df_all['lemmatized'] = df_all['tweet_text_cleaned'].apply(lambda x: lemmatize_tweet_text(x, allowed_postags=('NOUN', 'ADJ', 'VERB', 'ADV')))


# + tags=[]
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words |= {"attach", "ahead", "rt",'www','tinyurl','com', 'https', 'http',
               '&amp','amp', 'rt', 'bit', 'ly', 'bitly', 'trump', 'byte', 'bytes',
               'donald','emoji','earthquake','fire','flood','hurricane','wildfire','people','eqnz','water',
               'mexico','forest','greecefire','harvey','bahama','keralaflood','lanka','florence','dorian','irma',
               'ecuador','italian','terremoto','azad','mirpur','kaikoura','indian','islamabad','northern','woolsey',
               'greece','southern','malibu','ymmfire','chico','chengannur','lakh','donate lakh','onam','floridian','maria',
               'mozambique','zimbabwe','nebraska','matthew','haiti','crore','greek'}

# + tags=[]
# Build Disaster Type LDA Model
disaster_types = ['earthquake','fire', 'flood', 'hurricane']

# + tags=[]
vectorizer_params = {'analyzer' : 'word',
                     'min_df' : 10,
                     'stop_words' : stop_words,
                     'lowercase' : False,
                     'ngram_range' : (1,2)}

# + tags=[]
vectorizer = CountVectorizer(**vectorizer_params).fit(list(df_all['lemmatized']))

# + tags=[]
with open(str(product['vectorizer']), "wb") as f:
    pickle.dump(vectorizer, f)
# + tags=[]


