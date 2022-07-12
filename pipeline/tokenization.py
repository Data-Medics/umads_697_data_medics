# +
import pandas as pd
import numpy as np
import re

# NLP Imports
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split 
from tqdm import tqdm # Used to show a progress bar
import spacy

# Stemming in case we need it
import nltk
from nltk.stem.porter import PorterStemmer

# + tags=["parameters"]
upstream = []
random_seed = None
# -

# ## Read the train, dev,  test data, also the latest samples from Twitter

# +
df_dev = pd.read_csv('../data/HumAID_data_v1.0/all_combined/all_dev.tsv', sep='\t')
df_train = pd.read_csv('../data/HumAID_data_v1.0/all_combined/all_train.tsv', sep='\t')
df_test = pd.read_csv('../data/HumAID_data_v1.0/all_combined/all_test.tsv', sep='\t')

df_dev.dropna(inplace=True)
df_train.dropna(inplace=True)
df_test.dropna(inplace=True)

df_train.sample(5)
# -
# ## Build the tokenizer

# +
# Follow https://machinelearningknowledge.ai/complete-guide-to-spacy-tokenizer-with-examples/

# #!python -m spacy download en_core_web_sm

nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()

# Add more stop words if needed - https://machinelearningknowledge.ai/tutorial-for-stopwords-in-spacy/
nlp.Defaults.stop_words |= {"attach", "ahead", "rt"}

def tokenize(text):
    # Tokenize
    doc = nlp(text.lower())
    
    # Lematize, stop words and lematization removal
    tokens = [token.lemma_ for token in doc if not (token.is_stop or token.is_punct or token.is_digit)]
    
    # Stemming - uncomment to try it out
    #tokens = [stemmer.stem(token) for token in tokens]
    
    # Check more here - noun chunks, entities etc - https://stackabuse.com/python-for-nlp-tokenization-stemming-and-lemmatization-with-spacy-library/
    
    # Return the tokens
    return tokens

tokenize('RT @HotshotWake: Good morning from Stewart Crossing Yukon up in Canada. Big fire day ahead. #canada #yukon #wildfire https://t.co/cSoymOMwJO')
# -

# vectorizer = TfidfVectorizer(
#     tokenizer=tokenize,
#     strip_accents='unicode',
#     ngram_range=(1, 2),
#     max_features=10000,
#     use_idf=True
# )
vectorizer = TfidfVectorizer(tokenizer=tokenize, min_df=10, ngram_range=(1,2))

# %%time
X_train = vectorizer.fit_transform(df_train['tweet_text'])
y_train = list(df_train['class_label'])
X_train.shape

# Prepate the logistic regression classifier
clf = LogisticRegression(solver='lbfgs', multi_class='auto', random_state=random_seed, max_iter=1000)
clf.fit(X_train, y_train)

# +
# %%time
# Predict on test, must be about the same
X_test = vectorizer.transform(df_test['tweet_text'])
y_test = list(df_test['class_label'])

lr_test_preds = clf.predict(X_test)
# -

# Score on the test data
lr_f1 = f1_score(y_test, lr_test_preds, average='macro')
print(lr_f1)

vectorizer.vocabulary_


