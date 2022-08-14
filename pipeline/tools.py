import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
import tweepy
import datetime as dt
import spacy
import altair as alt
import networkx as nx
import nx_altair as nxa
import matplotlib.pyplot as plt
import seaborn as sns


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


def extract_assign_location(df_disaster_tweets, number_most_common=10):
    nlp = spacy.load('xx_ent_wiki_sm')

    # Go through the dev data and collect all the locations
    locations = []

    for _, row in tqdm(df_disaster_tweets.iterrows()):
        doc = nlp(row['tweet_text'])
        locations.append([ent.text for ent in doc.ents if ent.label_ in ['LOC']])

    df_disaster_tweets['location'] = locations

    locations_flatten = sum(df_disaster_tweets['location'].tolist(), [])
    locations_count = Counter(locations_flatten).most_common(number_most_common)

    df_locs_most_common = pd.DataFrame(locations_count, columns=['location', 'count'])

    return df_disaster_tweets, df_locs_most_common


def extract_assign_location_per_kind(df_disaster_tweets, disaster_kinds, number_most_common=10):
    nlp = spacy.load('xx_ent_wiki_sm')

    # Go through the dev data and collect all the locations
    locations = []

    for _, row in tqdm(df_disaster_tweets.iterrows()):
        doc = nlp(row['tweet_text'])
        locations.append([ent.text for ent in doc.ents if ent.label_ in ['LOC']])

    df_disaster_tweets['location'] = locations

    df_locs_most_common = None

    for kind in disaster_kinds:
        _df_disaster_tweets = df_disaster_tweets[df_disaster_tweets['disaster_kind'] == kind]
        locations_flatten = sum(df_disaster_tweets['location'].tolist(), [])
        locations_count = Counter(locations_flatten).most_common(number_most_common)
        _df = pd.DataFrame(locations_count, columns=['location', 'count'])
        _df['disaster_kind'] = kind

        if df_locs_most_common is None:
            df_locs_most_common = _df
        else:
            df_locs_most_common = df_locs_most_common.append(_df, ignore_index=True)

    return df_disaster_tweets, df_locs_most_common


def plot_class_boxplot(df, class_label=None, title=None):
    plt.figure(figsize=(10, 6))
    if class_label:
        df = df[df['class_label'] == class_label]
    ax = sns.boxplot(x="disaster_kind", y="count", data=df).set(title=title)

    _title = title if title else class_label if class_label else ''

    ax = sns.boxplot(x="disaster_kind", y="count", data=df).set(title=_title)


def disaster_title(disaster_kind, plural=False):
    if disaster_kind == 'recent_tweets_wildfire':
        return 'Wildfires' if plural else 'Wildfire'
    elif disaster_kind == 'recent_tweets_earthquake':
        return 'Earthquakes' if plural else 'Earthquake'
    elif disaster_kind == 'recent_tweets_flood':
        return 'Floods' if plural else 'Flood'
    elif disaster_kind == 'recent_tweets_hurricane':
        return 'Hurricanes' if plural else 'Hurricane'
    else:
        return 'Undefined'


def topNlocations(df, n=3):
    total = sum(df['count'])
    df2 = df[['location', 'count']].sort_values(by='count', ascending=False)
    return sorted(list(df2['location'])[:n]), total


def top_locations(df, loc_list, topN=3):
    df = df.copy()
    df['created_at'] = df['created_at'].dt.floor('2H')
    df = df.explode('location').reset_index(drop=True)

    # Leave just the top locations, since the rest are too much noise
    if loc_list:
        df = df[df['location'].isin(loc_list)]

    df.dropna(inplace=True)
    df['count'] = 1

    df = df[['created_at', 'class_label', 'location', 'count']].groupby(
        ['created_at', 'class_label', 'location']
    ).count().reset_index()

    df = df.groupby(['created_at', 'class_label']).apply(lambda a: topNlocations(a, topN)).reset_index()
    df = df.rename(columns={0: 'tmp'})
    df['locations'] = df.apply(lambda a: a['tmp'][0], axis=1)
    df['count'] = df.apply(lambda a: a['tmp'][1], axis=1)

    return df[['created_at', 'class_label', 'locations', 'count']]


def sampleNtweets(df, loc_list, n=20):
    # df = df[df['location'].isin(loc_list)]
    df = df.drop_duplicates(subset=['tweet_text'])
    if n < len(df):
        df = df.sample(n)
    return list(df['tweet_text'])


def top_locations_tweets(df, loc_list, topN=3, sample_tweets=10):
    df = df.copy()
    # TODO: Filter df by locations
    # Get the top locations
    df_locs = top_locations(df, loc_list)
    df['created_at'] = df['created_at'].dt.floor('2H')

    df = df[['created_at', 'class_label', 'location', 'tweet_text']].groupby(
        ['created_at', 'class_label']
    ).apply(lambda a: sampleNtweets(a, loc_list, sample_tweets)).reset_index()

    df = pd.merge(df, df_locs, on=['created_at', 'class_label'])
    df = df.rename(columns={0: 'tweet_text'})

    return df


def show_top_locations(df, class_labels, disaster_kind):
    source = df[df['class_label'].isin(class_labels)]

    # Brush for selection
    brush = alt.selection_interval(empty='none')

    chart = alt.Chart(source).mark_line(size=2).encode(
        x='created_at:T',
        y='count:Q',
        color='class_label:N',
        tooltip=['class_label:N', 'locations', 'count', 'created_at']
    ).add_selection(brush)

    ranked_text = alt.Chart(source).mark_text(align='left', dx=-250, dy=-100).encode(
        y=alt.Y('row_number:O',axis=None)
    ).transform_window(
        row_number='row_number()'
    ).transform_filter(
        brush
    ).transform_window(
        rank='rank(row_number)'
    ).transform_filter(
        alt.datum.rank<20
    )

    tweet_text = ranked_text.encode(text='tweet_text:N').properties(
        width=500,
        height=200
    ).properties(
        title='Tweets from the selected time period'
    )

    chart = chart.properties(
        width=500,
        height=400
    ).properties(
        title=disaster_title(disaster_kind) + ' disasters in the world, interactive chart'
    )

    return alt.vconcat(chart, tweet_text)


def get_top_locations_flatten(df, top_count=20):
    locations_flatten = sum(df['location'].tolist(), [])
    locations_count = Counter(locations_flatten).most_common(20)
    df_locs_most_common = pd.DataFrame(locations_count, columns=['location', 'count'])
    return list(df_locs_most_common['location'])


def get_location_graph_data(df, locations):
    df = df.copy()
    df = df.explode('location')
    df = df[df['location'].isin(locations)]
    df = df.dropna()

    df_loc_graph = pd.merge(df[['class_label', 'location']], df[['class_label', 'location']],
             left_index=True, right_index=True,
             how="inner")

    # Remove any duplicates
    df_loc_graph = df_loc_graph[df_loc_graph['location_x'] > df_loc_graph['location_y']]

    df_loc_graph = df_loc_graph.rename(columns={'class_label_x': 'class_label'})
    df_loc_graph = df_loc_graph[['class_label', 'location_x', 'location_y']]
    df_loc_graph = df_loc_graph[df_loc_graph['location_x'] != df_loc_graph['location_y']]

    df_loc_graph['count'] = 1

    df_loc_graph = df_loc_graph.groupby(by=['class_label', 'location_x', 'location_y']).count().reset_index()
    return df_loc_graph


def build_network(df):
    toret = nx.Graph()
    for row in df.iterrows():
        row = row[1]
        if (row['location_x'] not in toret.nodes):
            toret.add_node(row['location_x'])
            toret.nodes[row['location_x']]['count'] = 0
            toret.nodes[row['location_x']]['label'] = row['location_x']
        if (row['location_y'] not in toret.nodes):
            toret.add_node(row['location_y'])
            toret.nodes[row['location_y']]['count'] = 0
            toret.nodes[row['location_y']]['label'] = row['location_y']
        toret.nodes[row['location_x']]['count'] = toret.nodes[row['location_x']]['count'] + row['count']
        toret.nodes[row['location_y']]['count'] = toret.nodes[row['location_y']]['count'] + row['count']
        toret.add_edge(row['location_x'], row['location_y'])
        toret.edges[row['location_x'], row['location_y']]['count'] = int(row['count'])
    return toret


def get_layout(positions):
    # helper function to build a dataframe of positions for nodes
    elems = []
    nodes = list(positions.keys())
    for n in nodes:
        elems.append({'node': n, 'x': positions[n][0], 'y': positions[n][1]})
    return (pd.DataFrame(elems))


def plot_disaster_mentions(df, class_label, title):
    df = df[df['class_label'] == class_label]
    if df.shape[0] < 1:
        return 'No records found for class ' + class_label

    network = build_network(df)
    pos = nx.kamada_kawai_layout(network)

    e = nxa.draw_networkx_edges(network, pos=pos, width='weight:N')  # get the edge layer
    n = nxa.draw_networkx_nodes(network, pos=pos)  # get the node layer

    t = n.mark_text(dx=0, dy=20, color='black').encode(
        text=alt.Text('label:N')
    )

    n = n.mark_circle().encode(
        color=alt.Color('count:Q'),
        size=alt.Size('count:Q', scale=alt.Scale(range=[30, 500])),
        tooltip=['label:N', 'count:Q']
    )

    e = e.encode(
        color=alt.Color('count:Q', legend=None),
        tooltip=['count:Q']
    )

    return (t + e + n).properties(width=500, height=500).properties(
        title=title
    ).interactive()


def is_intersect(loc, locations):
    return set(locations).issubset(set(loc))


def show_tweets_locations(df, class_label, locations, sample=None):
    if not locations:
        return 'No location pairs provided'

    df = df.copy()
    df = df[df['class_label'] == class_label]
    if df.shape[0] < 1:
        return 'No records found for class ' + class_label

    df['show'] = df.apply(lambda row: is_intersect(row['location'], locations), axis=1)
    df = df[df['show']]
    if sample and sample <= df.shape[0]:
        return df[['tweet_text', 'location']].sample(sample).reset_index(drop=True)
    else:
        return df[['tweet_text', 'location']].reset_index(drop=True)


def top_location_pairs(df, class_label, top_pairs_count=1):
    df = df.copy()
    df = df[df['class_label'] == class_label]
    if df.shape[0] < 1:
        return [[]]

    df = df.sort_values(by='count', ascending=False)
    if top_pairs_count > df.shape[0]:
        return df[['location_x','location_y']].to_numpy().tolist()
    else:
        return df.head(top_pairs_count)[['location_x','location_y']].to_numpy().tolist()