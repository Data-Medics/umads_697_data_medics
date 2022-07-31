# # Building tweet vectorizer using a standard TweetTokenizer

# +
import pandas as pd
from sklearn.metrics import f1_score
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer
from sklearn.linear_model import LogisticRegression

# + tags=["parameters"]
upstream = []
random_seed = 42
# -

# ## Read the train and test data

# +
df_train = pd.read_csv('../data/HumAID_data_v1.0/all_combined/all_train.tsv', sep='\t')
df_test = pd.read_csv('../data/HumAID_data_v1.0/all_combined/all_test.tsv', sep='\t')

df_train.dropna(inplace=True)
df_test.dropna(inplace=True)

df_train.sample(5)
# -