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
import tweepy
import yaml
import datetime as dt
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

import matplotlib.pyplot as plt
# %matplotlib inline

random_seed = 10

#tags= ["parameters"]
upstream = []
product = None
query = None
language = None
limit=None
skip_interval_hours=None
credentials_file=None
random_seed=None


# +
### Identify - what type of disasters are in a block of tweets
# Load the disaster files
# Earthquake
df_earthquake_dev = pd.read_csv('../data/HumAID_data_v1.0/event_type/earthquake_dev.tsv', sep='\t')
df_earthquake_train = pd.read_csv('../data/HumAID_data_v1.0/event_type/earthquake_train.tsv', sep='\t')
df_earthquake_test = pd.read_csv('../data/HumAID_data_v1.0/event_type/earthquake_test.tsv', sep='\t')

df_earthquake_dev['disaster_type'] = 'earthquake'
df_earthquake_train['disaster_type'] = 'earthquake'
df_earthquake_test['disaster_type'] = 'earthquake'

df_earthquake_all = pd.concat([df_earthquake_dev, df_earthquake_train, df_earthquake_test])

# Fire
df_fire_dev = pd.read_csv('../data/HumAID_data_v1.0/event_type/fire_dev.tsv', sep='\t')
df_fire_train = pd.read_csv('../data/HumAID_data_v1.0/event_type/fire_train.tsv', sep='\t')
df_fire_test = pd.read_csv('../data/HumAID_data_v1.0/event_type/fire_test.tsv', sep='\t')

df_fire_dev['disaster_type'] = 'fire'
df_fire_train['disaster_type'] = 'fire'
df_fire_test['disaster_type'] = 'fire'

df_fire_all = pd.concat([df_fire_dev, df_fire_train, df_fire_test])

# Flood
df_flood_dev = pd.read_csv('../data/HumAID_data_v1.0/event_type/flood_dev.tsv', sep='\t')
df_flood_train = pd.read_csv('../data/HumAID_data_v1.0/event_type/flood_train.tsv', sep='\t')
df_flood_test = pd.read_csv('../data/HumAID_data_v1.0/event_type/flood_test.tsv', sep='\t')

df_flood_dev['disaster_type'] = 'flood'
df_flood_train['disaster_type'] = 'flood'
df_flood_test['disaster_type'] = 'flood'

df_flood_all = pd.concat([df_flood_dev, df_flood_train, df_flood_test])

# Hurricane
df_hurricane_dev = pd.read_csv('../data/HumAID_data_v1.0/event_type/hurricane_dev.tsv', sep='\t')
df_hurricane_train = pd.read_csv('../data/HumAID_data_v1.0/event_type/hurricane_train.tsv', sep='\t')
df_hurricane_test = pd.read_csv('../data/HumAID_data_v1.0/event_type/hurricane_test.tsv', sep='\t')

df_hurricane_dev['disaster_type'] = 'hurricane'
df_hurricane_train['disaster_type'] = 'hurricane'
df_hurricane_test['disaster_type'] = 'hurricane'

df_hurricane_all = pd.concat([df_hurricane_dev, df_hurricane_train, df_hurricane_test])

df_disaster_dev = pd.concat([df_earthquake_dev, df_fire_dev, df_flood_dev, df_hurricane_dev])
df_disaster_train = pd.concat([df_earthquake_train, df_fire_train, df_flood_train, df_hurricane_train])
df_disaster_test = pd.concat([df_earthquake_test, df_fire_test, df_flood_test, df_hurricane_test])

df_all = pd.concat([df_disaster_dev, df_disaster_train, df_disaster_test])

# -

df_all.head()



# ## clean, tokenize text

def text_pre_processing(x):
    # remove twitter handles (@user)
    text = re.sub(r'@[\S]*','',x)
    regex = re.compile('[^a-zA-Z]')
    text = regex.sub(' ', text)
    # remove web address
    text = re.sub(r'http[^\\S]*','',text)
#     # remove special characters, punctuations
#     text = text.replace(r'[^a-zA-Z#]', " ")
    # remove numbers
    text = text.replace(r'[0-9]', " ")
    # remove hashtag
    text = re.sub(r'#[\S]*','',text)
#    split compound words
    text = re.sub( r"([A-Z])", r" \1", text)
#     #remove non-ascii
    text = text.replace(r'[^\x00-\x7F]+', " ")
    #remove short words
    text = ' '.join([w for w in text.split() if len(w)>=3])
    #lowercase
    text = text.lower()
    #tokenize
    text = word_tokenize(text)
    return text
df_all['tweet_text_cleaned'] = df_all['tweet_text'].apply(lambda x: text_pre_processing(x))
df_all.head()


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
df_all.iloc[0]

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


# ## Lemmatizer

# +
def lemmatize(tweet, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    tokenized = []
    doc = nlp(' '.join(tweet))
    tokenized.append(' '.join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
#     tokenized.append(' '.join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc])) #if token.pos_ in allowed_postags]))
    return ''.join(tokenized)

# Do lemmatization keeping only Noun, Adj, Verb, Adverb
df_all['lemmatized'] = df_all['tweet_text_cleaned'].apply(lambda x: lemmatize(x, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']))
# -



# ## Train, Test, Split

X_train, X_test, y_train, y_test = train_test_split(df_all['lemmatized'], df_all['disaster_type'], test_size=0.25, random_state=random_seed)

# ## Bigram vectorizer

# Try predicting the disaster type of a corpus of tweets
bigram_vectorizer = TfidfVectorizer(stop_words='english', min_df=3, ngram_range=(1,2))
X_train_vectorized = bigram_vectorizer.fit_transform(X_train)
X_train_vectorized.shape

# ## Train Model

clf = LogisticRegression(solver='lbfgs', multi_class='auto', random_state=random_seed, max_iter=1000)
clf.fit(X_train_vectorized, y_train)


# ## Predict on test set

##transform and predict on test set
X_test_vectorized = bigram_vectorizer.transform(X_test)
y_pred_lr = clf.predict(X_test_vectorized)

lr_f1 = f1_score(y_test, y_pred_lr, average='macro')
print(lr_f1)

test_results_df = pd.DataFrame(list(zip(X_test,y_test,y_pred_lr)),columns=[['X_test','y_test','y_pred_lr']])

test_results_df.head(20)

# ## Confusion Matrix

fig = plt.figure(figsize=(15,6))
skplt.metrics.plot_confusion_matrix(y_test, y_pred_lr,
                                    title="Confusion Matrix",
                                    cmap="Oranges")

# +
import seaborn as sns
cf_matrix = confusion_matrix(y_test, y_pred_lr)

ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')
ax.set_title('Logistic Regression Confusion Matrix');
ax.set_xlabel('\nPredicted Disaster Type')
ax.set_ylabel('Actual Disaster Type ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['earthquake','fire', 'flood', 'hurricane'])
ax.yaxis.set_ticklabels(['earthquake','fire', 'flood', 'hurricane'])

## Display the visualization of the Confusion Matrix.
plt.show()
# -

# ## Precision / Recall Curve

plt.rcParams["figure.figsize"] = (20,10)
y_pred_proba_lr = clf.predict_proba(X_test_vectorized)
fig = plt.figure(figsize=(20,10))
skplt.metrics.plot_precision_recall_curve(y_test, y_pred_proba_lr)





# ## Get Random Disaster Tweets

# +
with open('credentials_file.yaml', 'r') as stream:
    credentials = yaml.safe_load(stream)

bearer_token = credentials['twitter_bearer_token']


# -

# Ref: https://docs.tweepy.org/en/latest/client.html#tweepy.Client.search_recent_tweets
#.     https://developer.twitter.com/en/docs/twitter-api/tweets/search/api-reference/get-tweets-search-recent
#.     https://dev.to/twitterdev/a-comprehensive-guide-for-using-the-twitter-api-v2-using-tweepy-in-python-15d9
#.     https://developer.twitter.com/en/docs/twitter-api/fields
#.     https://developer.twitter.com/en/docs/twitter-api/tweets/search/integrate/build-a-query#list
#.     https://developer.twitter.com/en/docs/twitter-api/tweets/filtered-stream/integrate/build-a-rule
def query_tweets(bearer_token, query, limit=1000, end_time=None):
    client = tweepy.Client(
        bearer_token=bearer_token,
        wait_on_rate_limit=True
    )

    tweets_collection = []
    tweet_fields=['text', 'created_at', 'entities', 'author_id']
    counter = 0

    for tweet in tweepy.Paginator(client.search_recent_tweets, query=query,
                                  tweet_fields=['text', 'created_at', 'entities'],
                                  expansions=['author_id', 'entities.mentions.username'],
                                  user_fields=['username'],
                                  max_results=100, end_time=end_time).flatten(limit=limit):
        #print(tweet.entities)
        hashtags = []
        users = []
        if tweet.entities:
            ht = tweet.entities.get('hashtags')
            if ht:
                hashtags = [tag.get('tag') for tag in ht]
                
            mentions = tweet.entities.get('mentions')
            if mentions:
                users = [mention.get('username') for mention in mentions]


        tweets_collection.append(
            {
                "author_id": tweet.author_id,
                "tweet_text": tweet.text,
                "created_at": pd.to_datetime(tweet.created_at),
                "hashtags": hashtags,
                "users": users
            }
        )
    
    return pd.DataFrame(tweets_collection)

# +
# %%time
skip_interval_hours = 2
language = 'en'
query = 'earthquake OR fire OR flood OR hurricane'
# We cannot query for more than a week back - the API returns an error
periods_one_week = int((24 * 6) / skip_interval_hours)
period_end = dt.datetime.now()
period_delta = dt.timedelta(hours=skip_interval_hours)

df_tweets = None

for _ in tqdm(range(periods_one_week)):
    df = query_tweets(bearer_token, query=query + ' lang:' + language, limit=2000, end_time=period_end)
    if df_tweets is None:
        df_tweets = df
    else:
        df_tweets = pd.concat([df_tweets, df])
    period_end -= period_delta
# -

df_tweets['created_at'] = pd.to_datetime(df_tweets['created_at'])
df_tweets.sample(10)





src_file_path = os.path.dirname(os.path.abspath("__file__"))
filename_random_tweets = os.path.join(src_file_path, 'output\\random_disaster_tweets.csv')
df_tweets.to_csv(filename_random_tweets)

df_random = pd.read_csv('output\\random_disaster_tweets.csv')
df_random.head()


df_random.head(100)

# ## Predict

df_random['tweet_text_cleaned'] = df_random['tweet_text'].apply(lambda x: text_pre_processing(x))
df_random['lemmatized'] = df_random['tweet_text_cleaned'].apply(lambda x: lemmatize(x, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']))
random_tweets_vectorized = bigram_vectorizer.transform(df_random['lemmatized'].tolist())
preds = clf.predict(random_tweets_vectorized)

combined = pd.DataFrame(list(zip(df_random['tweet_text'],preds)))
combined.head(200)
