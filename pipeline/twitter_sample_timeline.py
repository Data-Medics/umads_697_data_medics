import pandas as pd
import tweepy
import yaml
import datetime as dt
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# + tags=["parameters"]
upstream = []
product = None
query = None
language = None
limit=None
skip_interval_hours=None
credentials_file=None
random_seed=None
# -

# ## Load Twitter credentials

# +
with open(credentials_file, 'r') as stream:
    credentials = yaml.safe_load(stream)

bearer_token = credentials['twitter_bearer_token']


# -

# ## Sample data from Twitter

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

# ## Iterate through the past one week using the skip period

# +
# %%time
# We cannot query for more than a week back - the API returns an error
periods_one_week = int((24 * 6) / skip_interval_hours)
period_end = dt.datetime.now()
period_delta = dt.timedelta(hours=skip_interval_hours)

df_all = None

for _ in tqdm(range(periods_one_week)):
    df = query_tweets(bearer_token, query=query + ' lang:' + language, limit=limit, end_time=period_end)
    if df_all is None:
        df_all = df
    else:
        df_all = pd.concat([df_all, df])
    period_end -= period_delta
# -

df_all['created_at'] = pd.to_datetime(df_all['created_at'])
df_all.sample(10)

df_all.to_csv(product['file'], index=False)

# ## Clasify the tweets using the linear regression algorithm

# +
df_dev = pd.read_csv('../data/HumAID_data_v1.0/all_combined/all_dev.tsv', sep='\t')
df_train = pd.read_csv('../data/HumAID_data_v1.0/all_combined/all_train.tsv', sep='\t')
df_test = pd.read_csv('../data/HumAID_data_v1.0/all_combined/all_test.tsv', sep='\t')

df_dev.dropna(inplace=True)
df_train.dropna(inplace=True)
df_test.dropna(inplace=True)

df_train.sample(5)

# +
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)

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
# %%time
X_train = vectorizer.fit_transform(df_train['tweet_text'])
y_train = list(df_train['class_label'])

X_test = vectorizer.transform(df_test['tweet_text'])
y_test = list(df_test['class_label'])

X_train.shape
# -

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

# +
# %%time
X_sample = vectorizer.fit_transform(df_all['tweet_text'])
y_sample = clf.predict(X_sample)

df_all['class_label'] = y_sample
df_all.sample(20)
# -



# ## Extract the locations, pick the one or too most popular locations

import spacy
from spacy import displacy 

nlp = spacy.load('xx_ent_wiki_sm')

# +
# %%time
# Go through the dev data and collect all the locations
locations = []

for _, row in tqdm(df_all.iterrows()):
    doc = nlp(row['tweet_text'])
    locations.append([ent.text for ent in doc.ents if ent.label_ in ['LOC']])

# -

df_all['location'] = locations
df_all.sample(20)

# +
# %%time
locations_flatten = sum(df_all['location'].tolist(), [])

locations_count = Counter(locations_flatten).most_common(10)

df_locs_most_common = pd.DataFrame(locations_count, columns=['location', 'count'])

df_locs_most_common

# +
# TODO: Find which category is most predictive about the location
# -

# ## Plot the intensity for each catagory, and for a particular location

# +
df_plot = df_all.copy()

df_plot['created_at'] = df_plot['created_at'].dt.floor('8H') # Round to 8 hours
df_plot['count'] = 1
df_group_plot = df_plot[['created_at', 'class_label', 'count']].groupby(['created_at', 'class_label']).count().reset_index()
df_group_plot['created_at'] = pd.to_datetime(df_group_plot['created_at'])
df_group_plot.head()
#sns.lineplot(data=df_all.dropna(), x='created_at', y='class_label')
# -

df_group_plot['class_label'].unique()

# +
class_labels = ['displaced_people_and_evacuations',
                'injured_or_dead_people',
                'rescue_volunteering_or_donation_effort',
                'requests_or_urgent_needs',
                'sympathy_and_support',
                'infrastructure_and_utility_damage',
                'caution_and_advice']

plt.figure(figsize = (15,8))
sns.lineplot(data=df_group_plot[df_group_plot['class_label'].isin(class_labels)],
             x='created_at', y='count', hue='class_label');

# +
df_explode = df_all.copy()

most_common10 = df_locs_most_common.head(10)['location'].tolist()

df_explode = df_explode.explode('location').reset_index(drop=True)
df_explode.dropna(inplace=True)
df_explode['created_at'] = df_explode['created_at'].dt.floor('8H') # Round to 8 hours
df_explode['count'] = 1
df_group_explode = df_explode[['created_at', 'location', 'count']].groupby(
    ['created_at', 'location']
).count().reset_index()
df_group_explode = df_group_explode[df_group_explode['location'].isin(most_common10)]
df_group_explode.head()
# -

df_group_explode['location'].unique()

# +
locs = ['Portugal', 'Yosemite', 'Spain', 'Yosemite National Park']

plt.figure(figsize = (15,8))
sns.lineplot(data=df_group_explode[df_group_explode['location'].isin(locs)],
             x='created_at', y='count', hue='location');
# -




