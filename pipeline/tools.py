import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
import tweepy
import datetime as dt


def get_locations(nlp, df, limit_most_common=100):
    locations = []

    for _, row in df.iterrows():
        doc = nlp(row['tweet_text'])
        locations.extend([[ent.text] for ent in doc.ents if ent.label_ in ['LOC']])

    df_locs = pd.DataFrame(locations, columns=['Location'])

    locations = Counter(df_locs['Location']).most_common(limit_most_common)
    df_locs_most_common = pd.DataFrame(locations, columns=['location', 'count'])
    locations_set = {l.lower() for l in df_locs_most_common['location'] if not l.startswith('@')}

    return df_locs, locations_set


def generate_dense_features(tokenized_texts, model):
    result = []
    for tokens in tqdm(tokenized_texts):
        filtered_tokens = [t for t in tokens if t in model.wv.index_to_key]
        if len(filtered_tokens) > 0:
            vec = np.mean(model.wv[filtered_tokens], axis=0)
        else:
            vec = np.zeros(model.wv.vector_size)
        result.append(vec)
    return np.array(result)


class Tokenizer:
    def __init__(self, nlp, stopwords, stemmer=None):
        self.nlp = nlp
        self.stopwords = stopwords
        self.stemmer = stemmer
        self.nlp.Defaults.stop_words |= stopwords

    def tokenize(self, text):
        # Tokenize
        doc = self.nlp(text.lower())

        # Lematize, stop words and lematization removal
        tokens = [token.lemma_ for token in doc if not (token.is_stop or token.is_punct or token.is_digit)]

        # Stemming - uncomment to try it out
        if self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]

        # Check more here - noun chunks, entities etc - https://stackabuse.com/python-for-nlp-tokenization-stemming-and-lemmatization-with-spacy-library/

        # Return the tokens
        return tokens


# Ref: https://docs.tweepy.org/en/latest/client.html#tweepy.Client.search_recent_tweets
# .     https://developer.twitter.com/en/docs/twitter-api/tweets/search/api-reference/get-tweets-search-recent
# .     https://dev.to/twitterdev/a-comprehensive-guide-for-using-the-twitter-api-v2-using-tweepy-in-python-15d9
# .     https://developer.twitter.com/en/docs/twitter-api/fields
# .     https://developer.twitter.com/en/docs/twitter-api/tweets/search/integrate/build-a-query#list
# .     https://developer.twitter.com/en/docs/twitter-api/tweets/filtered-stream/integrate/build-a-rule
def query_tweets(bearer_token, query, limit=1000, end_time=None):
    client = tweepy.Client(
        bearer_token=bearer_token,
        wait_on_rate_limit=True
    )

    tweets_collection = []
    tweet_fields = ['text', 'created_at', 'entities', 'author_id']
    counter = 0

    for tweet in tweepy.Paginator(client.search_recent_tweets, query=query,
                                  tweet_fields=['text', 'created_at', 'entities'],
                                  expansions=['author_id', 'entities.mentions.username'],
                                  user_fields=['username'],
                                  max_results=100, end_time=end_time).flatten(limit=limit):
        # print(tweet.entities)
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


def retrieve_tweets(bearer_token, skip_interval_hours, query, language, limit):
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

    return df_all
