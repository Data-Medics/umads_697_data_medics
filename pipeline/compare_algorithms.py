# +
import pandas as pd
import numpy as np

# NLP Imports
from collections import Counter

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from tqdm import tqdm # Used to show a progress bar
import spacy
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB

# + tags=["parameters"]
upstream = ['tokenization']
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
# ## Read the already trained vectorizer

with open(str(upstream['tokenization']['vectorizer']), 'rb') as f:
    vectorizer = pickle.load(f)

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import WordNetLemmatizer, pos_tag, TreebankWordTokenizer
import re
import numpy as np

# +
# _tokernizer = TreebankWordTokenizer()
# _lem = WordNetLemmatizer()
# _pos_map = {'J': 'a', 'N': 'n', 'R': 'r', 'V': 'v'}

# +
# def tokenize_text(text):
#     # 1. Tokenise to alphabetic tokens
#     #     tokeniser = RegexpTokenizer(r'\b[A-Za-z]+\b')
#     #     tokeniser = RegexpTokenizer(r'\b\w\w+\b')
#     tokens = _tokernizer.tokenize(text)

#     # 1. POS tagging
#     pos_tags = pos_tag(tokens)

#     # 2. Lemmatise
#     lemmas = [_lem.lemmatize(
#         token,
#         pos=_pos_map.get(tag[0], 'v'))
#         for token, tag in pos_tags]

#     return lemmas
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)


# -

def get_vectorizer_custom():
    vectorizer = TfidfVectorizer(
        tokenizer=tokenizer.tokenize, #tokenize_text,
        strip_accents='unicode',
        ngram_range=(1, 2),
        max_df=0.90,
        min_df=1,
        max_features=10000,
        use_idf=True
    )
    return vectorizer


vectorizer = get_vectorizer_custom()

# +
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=500, n_iter=30, random_state=42)

# +
# %%time
X_train = vectorizer.fit_transform(df_train['tweet_text'])
X_train_reduced = svd.fit_transform(X_train)
y_train = list(df_train['class_label'])

X_test = vectorizer.transform(df_test['tweet_text'])
X_test_reduced = svd.transform(X_test)
y_test = list(df_test['class_label'])

X_train.shape
# -

# ## Test logistic regression

# %%time
# Prepate the logistic regression classifier
clf = LogisticRegression(solver='lbfgs', multi_class='auto', random_state=random_seed, max_iter=1000)
clf.fit(X_train, y_train)

# %%time
# Predict on test
lr_test_preds = clf.predict(X_test)
# Score on the test data
lr_f1 = f1_score(y_test, lr_test_preds, average='macro')
print(lr_f1)

# ## Test random forest

# %%time
clf_rf = RandomForestClassifier(n_estimators=100, max_depth=10)
clf_rf.fit(X_train_reduced, y_train)


# %%time
# Predict on test
rf_test_preds = clf_rf.predict(X_test_reduced)
# Score on the test data
rf_f1 = f1_score(y_test, rf_test_preds, average='macro')
print(rf_f1)

# ## Test Gradient Boosting Classifier - takes too long

# +
# # %%time
# clf_gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=random_seed)
# clf_gbc.fit(X_train, y_train)

# +
# # %%time
# # Predict on test
# gbc_test_preds = clf_gbc.predict(X_test)
# # Score on the test data
# gbc_f1 = f1_score(y_test, gbc_test_preds, average='macro')
# print(gbc_f1)
# -

# ## Test MultinomialNB

# %%time
clf_mnb = MultinomialNB()
clf_mnb.fit(X_train, y_train)

# %%time
# Predict on test
mnb_test_preds = clf_mnb.predict(X_test)
# Score on the test data
mnb_f1 = f1_score(y_test, mnb_test_preds, average='macro')
print(mnb_f1)

# ## Test Voting Classifier

# +
clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=42, max_iter=1000)
clf2 = RandomForestClassifier(n_estimators=50, random_state=42)
clf3 = MultinomialNB()

eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('mnb', clf3)], voting='soft')
eclf1.fit(X_train, y_train)
# -

# %%time
# Predict on test
ec_test_preds = eclf1.predict(X_test)
# Score on the test data
ec_f1 = f1_score(y_test, ec_test_preds, average='macro')
print(ec_f1)


