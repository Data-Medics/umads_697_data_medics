# # Test and compare different classification models, store the trained models for using downstream

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

from sklearn.model_selection import GridSearchCV

# + tags=["parameters"]
upstream = ['vectorizer']
random_seed = None
# -

# ## Read the train and test data

# +
df_train = pd.read_csv('../data/HumAID_data_v1.0/all_combined/all_train.tsv', sep='\t')
df_test = pd.read_csv('../data/HumAID_data_v1.0/all_combined/all_test.tsv', sep='\t')

df_train.dropna(inplace=True)
df_test.dropna(inplace=True)

df_train.sample(5)
# -
# ## Read the already trained vectorizer

with open(upstream["vectorizer"]["vectorizer"], 'rb') as f:
    vectorizer = pickle.load(f)

# +
# %%time
X_train = vectorizer.fit_transform(df_train['tweet_text'])
y_train = list(df_train['class_label'])

X_test = vectorizer.transform(df_test['tweet_text'])
y_test = list(df_test['class_label'])

X_train.shape
# -

# ## Logistic regression test

# +
# %%time
# Prepate the logistic regression classifier
#clf_lr = LogisticRegression(solver='lbfgs', multi_class='auto', random_state=random_seed, max_iter=1000)
clf_lr = LogisticRegression()

# parameters_lr = {'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
#                  'multi_class': ['auto'], 'max_iter': [500, 1000, 2000]}

parameters_lr = {'solver':['liblinear'],
                 'multi_class': ['auto'], 'max_iter': [500]}

clf = GridSearchCV(clf_lr, parameters_lr)

clf.fit(X_train, y_train)
# -

# Show the optimal parameters
clf.best_params_

# %%time
# Predict on test
lr_test_preds = clf.predict(X_test)
# Score on the test data
lr_f1 = f1_score(y_test, lr_test_preds, average='macro')
print(lr_f1)

# Store the model
with open(str(product['model_lr']), 'wb') as f:
    pickle.dump(clf, f)

# ## Random Forest test

# %%time
clf_rf = RandomForestClassifier(n_estimators=100, max_depth=10)
clf_rf.fit(X_train, y_train)

# %%time
# Predict on test
rf_test_preds = clf_rf.predict(X_test)
# Score on the test data
rf_f1 = f1_score(y_test, rf_test_preds, average='macro')
print(rf_f1)

# Store the model
with open(str(product['model_rf']), 'wb') as f:
    pickle.dump(clf_rf, f)

# ## Gradient Boosting Classifier - takes too long

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

# ## MultinomialNB test

# %%time
clf_mnb = MultinomialNB()
clf_mnb.fit(X_train, y_train)

# %%time
# Predict on test
mnb_test_preds = clf_mnb.predict(X_test)
# Score on the test data
mnb_f1 = f1_score(y_test, mnb_test_preds, average='macro')
print(mnb_f1)

# Store the model
with open(str(product['model_nb']), 'wb') as f:
    pickle.dump(clf_mnb, f)

# ## Voting Classifier test

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

# Store the model
with open(str(product['model_votingc']), 'wb') as f:
    pickle.dump(eclf1, f)


