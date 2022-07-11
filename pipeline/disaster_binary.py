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

# + tags=["parameters"]
upstream = ["twitter_random_sample", "twitter_wildfire_sample"]
random_seed = None
# -

# ## Read the train, dev,  test data, also the latest samples from Twetter

# +
df_dev = pd.read_csv('../data/HumAID_data_v1.0/all_combined/all_dev.tsv', sep='\t')
df_train = pd.read_csv('../data/HumAID_data_v1.0/all_combined/all_train.tsv', sep='\t')
df_test = pd.read_csv('../data/HumAID_data_v1.0/all_combined/all_test.tsv', sep='\t')

df_random_sample = pd.read_csv(upstream["twitter_random_sample"]["file"])
df_wildfire_sample = pd.read_csv(upstream["twitter_wildfire_sample"]["file"])

df_dev.dropna(inplace=True)
df_train.dropna(inplace=True)
df_test.dropna(inplace=True)

df_random_sample.dropna(inplace=True)
df_wildfire_sample.dropna(inplace=True)
# -
df_dev.sample(10)

df_random_sample.sample(10)

df_wildfire_sample.sample(10)

# Show the number of rows in the dev set
print("Dev rows:", df_dev.shape[0])
print("Train rows:", df_train.shape[0])
print("Test rows:", df_test.shape[0])
# Show the number of labels/categories
len(set(df_train['class_label']))


# Show the unique labels/categories
list(df_train['class_label'].unique())

# ## Prepare test and training set based on the labeled and the random tweets

# +
# Assign a binary label - disaster is 1, non-disaster is 0
df_train['disaster'] = 1
df_train.loc[df_train['class_label'] == 'not_humanitarian', 'disaster'] = 0

df_test['disaster'] = 1
df_test.loc[df_test['class_label'] == 'not_humanitarian', 'disaster'] = 0

print('Disaster tweets count: ', df_train[df_train['class_label'] != 'not_humanitarian'].shape[0])
print('Non-disaster tweets count: ', df_train[df_train['class_label'] == 'not_humanitarian'].shape[0])
df_train[df_train['class_label'] == 'not_humanitarian'].sample(5)


# +
# Split non-disasters to train and test
df_random_sample['disaster'] = 0

df_random_train = df_random_sample.sample(frac=0.8, random_state=random_seed) #random state is a seed value
df_random_test = df_random_sample.drop(df_random_train.index)

df_random_train.sample(5)

# +
# Prepare the final train and test dataframes
df_final_train = pd.concat(
    [
        df_train[['tweet_text', 'disaster']],
        df_random_train[['tweet_text', 'disaster']]
    ]
)

df_final_test = pd.concat(
    [
        df_test[['tweet_text', 'disaster']],
        df_random_test[['tweet_text', 'disaster']]
    ]
)


df_final_test.sample(5)
# -

# ## Vectorize and traing a model

# %%time
## Vectorize, prepate the training data and labels
bigram_vectorizer = TfidfVectorizer(stop_words='english', min_df=10, ngram_range=(1,2))
X_train = bigram_vectorizer.fit_transform(df_final_train['tweet_text'])
y_train = list(df_final_train['disaster'])
X_train.shape

# Prepate the logistic regression classifier
clf = LogisticRegression(solver='lbfgs', multi_class='auto', random_state=random_seed, max_iter=1000)
clf.fit(X_train, y_train)

# +
# Predict on test, must be about the same
X_test = bigram_vectorizer.transform(df_final_test['tweet_text'])
y_test = list(df_final_test['disaster'])

lr_test_preds = clf.predict(X_test)
# -

# Score on the test data
lr_f1 = f1_score(y_test, lr_test_preds, average='macro')
print(lr_f1)

# ## See how this model will predict on the wildfire data

df_wildfire_sample['disaster'] = 1
df_wildfire_sample.sample(5)

# +
X_wildfire_test = bigram_vectorizer.transform(df_wildfire_sample['tweet_text'])
y_wildfire_test = list(df_wildfire_sample['disaster'])

lr_wildfire_test_preds = clf.predict(X_wildfire_test)
# -

# Score on the test data
lr_f1 = f1_score(y_wildfire_test, lr_wildfire_test_preds, average='macro')
print(lr_f1)

# Check some of tweets that classified as disasters
disaster_idx = np.nonzero(lr_wildfire_test_preds == 1)
disaster_idx[0][:10]

nondisaster_idx = np.nonzero(lr_wildfire_test_preds == 0)
nondisaster_idx[0][:10]

df_wildfire_sample.iloc[disaster_idx].sample(10)

df_wildfire_sample.iloc[nondisaster_idx].sample(10)

df_wildfire_sample['tweet_text'][8917]


