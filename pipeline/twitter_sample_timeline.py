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
import altair as alt
import networkx as nx 
import nx_altair as nxa

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
#locs = ['Portugal', 'Yosemite', 'Spain', 'Yosemite National Park']
locs = ['Europe', 'France', 'Portugal', 'Spain', 'Birx', 'London', 'Athens']

plt.figure(figsize = (15,8))
sns.lineplot(data=df_group_explode[df_group_explode['location'].isin(locs)],
             x='created_at', y='count', hue='location');
# +
chart = alt.Chart(df_group_explode[df_group_explode['location'].isin(locs)]).mark_line(size=2).encode(
    x='created_at:T',
    y='count:Q',
    color='location:N',
    tooltip=['location', 'count']
)

chart.properties(
    width=500,
    height=500
).properties(
    title='Wildfire disasters in the world'
).interactive()


# -

# ### Try showing the category per location

# +
def category_per_location(df, location_list):
    df = df.copy()
    
    df = df.explode('location').reset_index(drop=True)
    df.dropna(inplace=True)
    
    df['created_at'] = df['created_at'].dt.floor('1H')
    
    df['count'] = 1
    
    df = df[['created_at', 'class_label', 'location', 'count']].groupby(
        ['created_at', 'class_label', 'location']
    ).count().reset_index()
    
    df = df[df['location'].isin(location_list)]
    df['location_class'] = df['location'] + ': ' + df['class_label']
    
    return df

def plot_category_per_location(df, location_list, class_labels=['displaced_people_and_evacuations',
                                                                'injured_or_dead_people',
                                                                'rescue_volunteering_or_donation_effort',
                                                                'requests_or_urgent_needs',
                                                                'sympathy_and_support',
                                                                'infrastructure_and_utility_damage',
                                                                'caution_and_advice']):
    df = category_per_location(df, location_list)
    
    plt.figure(figsize = (15,8))
    sns.lineplot(data=df[df['class_label'].isin(class_labels)],
             x='created_at', y='count', hue='location_class');
    

df_locations = category_per_location(df_all, ['Portugal', 'Spain'])
df_locations.sample(5)
# -

df_locations = category_per_location(df_all, ['Portugal'])
df_locations.sample(5)

plot_category_per_location(df_all, ['Portugal'])

plot_category_per_location(df_all, ['Portugal', 'Spain'], class_labels=['other_relevant_information'])

# ### Try combining the most important locations per category

# +
locs = ['Europe', 'France', 'Portugal', 'Spain', 'Birx', 'London', 'Athens']

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

df_top_locations = top_locations(df_all, locs)
df_top_locations

# +
class_labels = [
    'displaced_people_and_evacuations',
    'injured_or_dead_people',
    'rescue_volunteering_or_donation_effort',
    'requests_or_urgent_needs',
    'sympathy_and_support',
    'infrastructure_and_utility_damage',
    'caution_and_advice'
]

chart = alt.Chart(df_top_locations[df_top_locations['class_label'].isin(class_labels)]).mark_line(size=2).encode(
    x='created_at:T',
    y='count:Q',
    color='class_label:N',
    tooltip=['class_label:N', 'locations', 'count']
)

chart.properties(
    width=500,
    height=500
).properties(
    title='Wildfire disasters in the world'
)
# -

# ### Add sample tweets to the chart above

# +
# Add sample tweets to the chart above
## TODO: https://altair-viz.github.io/gallery/scatter_linked_table.html

locs = ['Europe', 'France', 'Portugal', 'Spain', 'Birx', 'London', 'Athens']

def sampleNtweets(df, loc_list, n=10):
    #df = df[df['location'].isin(loc_list)]
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

df_top_location_tweets = top_locations_tweets(df_all, locs)
df_top_location_tweets.sample(5)

# +
class_labels = [
    'displaced_people_and_evacuations',
    'injured_or_dead_people',
    'rescue_volunteering_or_donation_effort',
    'requests_or_urgent_needs',
    'sympathy_and_support',
    'infrastructure_and_utility_damage',
    'caution_and_advice'
]

source = df_top_location_tweets[df_top_location_tweets['class_label'].isin(class_labels)]

# Brush for selection
brush = alt.selection_interval(empty='none')

chart = alt.Chart(source).mark_line(size=2).encode(
    x='created_at:T',
    y='count:Q',
    color='class_label:N',
    tooltip=['class_label:N', 'locations', 'count']
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
    height=100
).properties(
    title='Tweet Text'
)

chart = chart.properties(
    width=500,
    height=500
).properties(
    title='Wildfire disasters in the world'
)

composite = alt.vconcat(chart, tweet_text)
composite
# -





# # Come up with a better visualization (using Altair)

# +
locations_flatten = sum(df_all['location'].tolist(), [])

locations_count = Counter(locations_flatten).most_common(20)

df_locs_most_common = pd.DataFrame(locations_count, columns=['location', 'count'])

loc20 = list(df_locs_most_common['location'])
loc20
# -

df_ = df_all.copy()
df_ = df_.explode('location')
df_ = df_[df_['location'].isin(loc20)]
df_ = df_.dropna()
print(df_.shape)
df_.head(5)

df_[df_['location'] == 'Birx'].iloc[1604]['tweet_text']

# +
df_loc_graph = pd.merge(df_[['class_label', 'location']], df_[['class_label', 'location']], 
         left_index=True, right_index=True,
         how="inner")

# Remove any duplicates
df_loc_graph = df_loc_graph[df_loc_graph['location_x'] > df_loc_graph['location_y']]

df_loc_graph = df_loc_graph.rename(columns={'class_label_x': 'class_label'})
df_loc_graph = df_loc_graph[['class_label', 'location_x', 'location_y']]
df_loc_graph = df_loc_graph[df_loc_graph['location_x'] != df_loc_graph['location_y']]

df_loc_graph['count'] = 1

df_loc_graph = df_loc_graph.groupby(by=['class_label', 'location_x', 'location_y']).count().reset_index()
#df_loc_graph = df_loc_graph[df_loc_graph['count'] > 10]
df_loc_graph.sample(20)


# +
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
        toret.add_edge(row['location_x'],row['location_y'])
        toret.edges[row['location_x'],row['location_y']]['count'] = int(row['count'])
    return toret

def get_layout(positions):
    # helper function to build a dataframe of positions for nodes 
    elems = []
    nodes = list(positions.keys())
    for n in nodes:
        elems.append({'node':n,'x':positions[n][0],'y':positions[n][1]}) 
    return(pd.DataFrame(elems))

def plot_disaster_mentions(df, class_label, title):
    network = build_network(df[df_loc_graph['class_label'] == class_label])
    pos = nx.kamada_kawai_layout(network)
    
    e = nxa.draw_networkx_edges(network, pos=pos, width='weight:N') # get the edge layer 
    n = nxa.draw_networkx_nodes(network, pos=pos) # get the node layer

    n = n.mark_circle().encode(
        color=alt.Color('count:Q'),
        size=alt.Size('count:Q', scale=alt.Scale(range=[30, 500])),
        tooltip=['label:N', 'count:Q']
    )

    e = e.encode(
        color=alt.Color('count:Q', legend=None),
        tooltip=['count:Q']
    )

    return (e+n).properties(width=500,height=500).properties(
        title=title
    ).interactive()

def is_intersect(loc, locations):
    return set(locations).issubset(set(loc))

def show_tweets_locations(df, class_label, locations, sample=None):
    df = df.copy()
    df = df[df['class_label'] == class_label]
    df['show'] = df.apply(lambda row: is_intersect(row['location'], locations), axis = 1)
    df = df[df['show']]
    if sample and sample <= df.shape[0]:
        return df[['tweet_text', 'location']].sample(sample).reset_index(drop=True)
    else:
        return df[['tweet_text', 'location']].reset_index(drop=True)


# -

pd.set_option('display.max_colwidth', 280)
show_tweets_locations(df_all, class_label='caution_and_advice' , locations=['Portugal', 'Athens'])

# +
# 'displaced_people_and_evacuations',
# 'injured_or_dead_people',
# 'rescue_volunteering_or_donation_effort',
# 'requests_or_urgent_needs',
# 'sympathy_and_support',
# 'infrastructure_and_utility_damage',
# 'caution_and_advice'
# -

plot_disaster_mentions(df_loc_graph, 'caution_and_advice', 'Caution And Advice')

show_tweets_locations(df_all, class_label='caution_and_advice',
                      locations=['Portugal'], sample=None)

plot_disaster_mentions(df_loc_graph, 'displaced_people_and_evacuations', 'Displaced People And Evacuations')

show_tweets_locations(df_all, class_label='displaced_people_and_evacuations',
                      locations=['California'])

plot_disaster_mentions(df_loc_graph, 'injured_or_dead_people', 'Injured Or Dead People')

show_tweets_locations(df_all, class_label='injured_or_dead_people',
                      locations=['Maroposa County'], sample=10)

plot_disaster_mentions(df_loc_graph, 'rescue_volunteering_or_donation_effort',
                       'Rescue Volunteering Or Donation Effort')

show_tweets_locations(df_all, class_label='rescue_volunteering_or_donation_effort',
                      locations=['Spain'], sample=10)

plot_disaster_mentions(df_loc_graph, 'requests_or_urgent_needs',
                       'requests_or_urgent_needs')

show_tweets_locations(df_all, class_label='requests_or_urgent_needs',
                      locations=['UK'], sample=10)

plot_disaster_mentions(df_loc_graph, 'sympathy_and_support',
                       'sympathy_and_support')

show_tweets_locations(df_all, class_label='sympathy_and_support',
                      locations=['Spain'], sample=10)

plot_disaster_mentions(df_loc_graph, 'infrastructure_and_utility_damage',
                       'infrastructure_and_utility_damage')

show_tweets_locations(df_all, class_label='infrastructure_and_utility_damage',
                      locations=['Darford'], sample=10)





# # Find the topics per disaster type, compare them, process them separately. look at the intensity





# ## Predict the disaster type from the sample

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

# Fire
df_fire_dev = pd.read_csv('../data/HumAID_data_v1.0/event_type/fire_dev.tsv', sep='\t')
df_fire_train = pd.read_csv('../data/HumAID_data_v1.0/event_type/fire_train.tsv', sep='\t')
df_fire_test = pd.read_csv('../data/HumAID_data_v1.0/event_type/fire_test.tsv', sep='\t')

df_fire_dev['disaster_type'] = 'fire'
df_fire_train['disaster_type'] = 'fire'
df_fire_test['disaster_type'] = 'fire'

# Flood
df_flood_dev = pd.read_csv('../data/HumAID_data_v1.0/event_type/flood_dev.tsv', sep='\t')
df_flood_train = pd.read_csv('../data/HumAID_data_v1.0/event_type/flood_train.tsv', sep='\t')
df_flood_test = pd.read_csv('../data/HumAID_data_v1.0/event_type/flood_test.tsv', sep='\t')

df_flood_dev['disaster_type'] = 'flood'
df_flood_train['disaster_type'] = 'flood'
df_flood_test['disaster_type'] = 'flood'

# Hurricane
df_hurricane_dev = pd.read_csv('../data/HumAID_data_v1.0/event_type/hurricane_dev.tsv', sep='\t')
df_hurricane_train = pd.read_csv('../data/HumAID_data_v1.0/event_type/hurricane_train.tsv', sep='\t')
df_hurricane_test = pd.read_csv('../data/HumAID_data_v1.0/event_type/hurricane_test.tsv', sep='\t')

df_hurricane_dev['disaster_type'] = 'hurricane'
df_hurricane_train['disaster_type'] = 'hurricane'
df_hurricane_test['disaster_type'] = 'hurricane'

df_disaster_dev = pd.concat([df_earthquake_dev, df_fire_dev, df_flood_dev, df_hurricane_dev])
df_disaster_train = pd.concat([df_earthquake_train, df_fire_train, df_flood_train, df_hurricane_train])
df_disaster_test = pd.concat([df_earthquake_test, df_fire_test, df_flood_test, df_hurricane_test])
# -


df_disaster_train.sample(5)

Counter(df_disaster_train['disaster_type'])


# +
# Credit to https://rensdimmendaal.com/data-science/undersampling-with-pandas/
def downsample(df:pd.DataFrame, label_col_name:str) -> pd.DataFrame:
    # find the number of observations in the smallest group
    nmin = df[label_col_name].value_counts().min()
    return (df
            # split the dataframe per group
            .groupby(label_col_name)
            # sample nmin observations from each group
            .apply(lambda x: x.sample(nmin))
            # recombine the dataframes
            .reset_index(drop=True)
            )

df_disaster_train_downsample = downsample(df_disaster_train, 'disaster_type')
# -

Counter(df_disaster_train_downsample['disaster_type'])

vectorizer_type = get_vectorizer_custom()

# +
# %%time
X_disaster_train = vectorizer_type.fit_transform(df_disaster_train_downsample['tweet_text'])
y_disaster_train = list(df_disaster_train_downsample['disaster_type'])

X_disaster_test = vectorizer_type.transform(df_disaster_test['tweet_text'])
y_disaster_test = list(df_disaster_test['disaster_type'])

X_disaster_train.shape
# -

clf = LogisticRegression(solver='lbfgs', multi_class='auto', random_state=random_seed, max_iter=1000)
clf.fit(X_disaster_train, y_disaster_train)

# +
lr_test_preds = clf.predict(X_disaster_test)

lr_f1 = f1_score(y_disaster_test, lr_test_preds, average='macro')
print(lr_f1)
# -

# ### Work on the recent tweets sample - this approach looks useless, almost all is predicted as a hurricane

X_recent_sample = vectorizer_type.fit_transform(df_all['tweet_text'])

y_recent_pred = clf.predict(X_recent_sample)

df_all['disaster_type'] = y_recent_pred

print(df_all.shape)
df_all.sample(5)

Counter(df_all['disaster_type'])


