import pandas as pd
import tweepy
import yaml

# + tags=["parameters"]
upstream = []
product = None
query = None
language = None
limit=None
credentials_file=None
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
def query_tweets(bearer_token, query, limit=1000):
    client = tweepy.Client(
        bearer_token=bearer_token,
        wait_on_rate_limit=True
    )

    tweets_collection = []

    for tweet in tweepy.Paginator(client.search_recent_tweets, query=query,
                                  tweet_fields=['text', 'created_at', 'entities'],
                                  max_results=100).flatten(limit=limit):
        hashtags = []
        if tweet.entities:
            ht = tweet.entities.get('hashtags')
            if ht:
                hashtags = [tag.get('tag') for tag in ht]


        tweets_collection.append(
            {
                "tweet_text": tweet.text,
                "created_at": pd.to_datetime(tweet.created_at),
                "hashtags": " ".join(hashtags)
            }
        )
    
    return pd.DataFrame(tweets_collection)

# %%time
df = query_tweets(bearer_token, query=query + ' lang:' + language, limit=limit)
df.sample(10)

df.to_csv(product['file'], index=False)


