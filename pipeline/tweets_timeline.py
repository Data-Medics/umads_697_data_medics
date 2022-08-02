# # Doing an exploration on recent disaster tweets

# +
import pandas as pd
import pickle

import spacy
from spacy import displacy
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import networkx as nx 
import nx_altair as nxa
from collections import Counter

from tools import extract_assign_location

# + tags=["parameters"]
upstream = [
    'vectorizer',
    'category_classification_models',
    'recent_tweets_wildfire',
    'recent_tweets_earthquake',
    'recent_tweets_flood',
    'recent_tweets_hurricane'
]
random_seed = None
disaster_kind = None


# -

# ## Load the sample of recent tweets

# +
def tweets

df_tweets = pd.read_csv(upstream[disaster_kind]['file'])
df_tweets.sample(5)
# -

# ## Extract the locations from the tweet sample

# +
# %%time
df_tweets, df_locs_most_common = extract_assign_location(df_tweets, number_most_common=10)

df_tweets.sample(5)
# -

df_locs_most_common

# ## Predict the class labels

# Get the vectorizer
with open(upstream["vectorizer"]["vectorizer"], 'rb') as f:
    vectorizer = pickle.load(f)

# Get the classification model
with open(upstream["category_classification_models"]["model_lr"], 'rb') as f:
    classifier = pickle.load(f)

# +
# %%time
# Predict the class labels
class_label = classifier.predict(vectorizer.transform(df_tweets['tweet_text']))
df_tweets['class_label'] = class_label

df_tweets.sample(5)
# -

# ## Plot the intensity for each catagory, and for a particular location

# +
df_tweets['created_at'] = pd.to_datetime(df_tweets['created_at'])
df_plot = df_tweets.copy()

df_plot['created_at'] = df_plot['created_at'].dt.floor('8H') # Round to 8 hours
df_plot['count'] = 1
df_group_plot = df_plot[['created_at', 'class_label', 'count']].groupby(['created_at', 'class_label']).count().reset_index()
df_group_plot['created_at'] = pd.to_datetime(df_group_plot['created_at'])
df_group_plot.head()
# -

df_group_plot['class_label'].unique()

# +
df_explode = df_tweets.copy()

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

top10_locs = list(df_group_explode['location'].unique())
top10_locs

# +
# locs = ['California', 'California’s', 'Mariposa County', 'Oak Fire',
#        'Yosemite', 'Yosemite National Park', 'North Texas', 'Burgos',
#        'Calistoga Ranch', 'Quintanilla del Coco']

# plt.figure(figsize = (15,8))
# sns.lineplot(data=df_group_explode[df_group_explode['location'].isin(locs)],
#              x='created_at', y='count', hue='location');

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

# ## Combine the most important locatoions per category

# +
# locs = ['California', 'California’s', 'Mariposa County', 'Oak Fire',
#        'Yosemite', 'Yosemite National Park', 'North Texas', 'Burgos',
#        'Calistoga Ranch', 'Quintanilla del Coco']

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

df_top_locations = top_locations(df_tweets, top10_locs)
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

# ## Show sample tweets per most important locations above

# +
# locs = ['California', 'California’s', 'Mariposa County', 'Oak Fire',
#        'Yosemite', 'Yosemite National Park', 'North Texas', 'Burgos',
#        'Calistoga Ranch', 'Quintanilla del Coco']

def sampleNtweets(df, loc_list, n=20):
    #df = df[df['location'].isin(loc_list)]
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

df_top_location_tweets = top_locations_tweets(df_tweets, top10_locs)
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
    title='Wildfire disasters in the world'
)

composite = alt.vconcat(chart, tweet_text)
composite
# -

# ## Visualize the locations

# +
locations_flatten = sum(df_tweets['location'].tolist(), [])

locations_count = Counter(locations_flatten).most_common(20)

df_locs_most_common = pd.DataFrame(locations_count, columns=['location', 'count'])

loc20 = list(df_locs_most_common['location'])
loc20
# -

df_ = df_tweets.copy()
df_ = df_.explode('location')
df_ = df_[df_['location'].isin(loc20)]
df_ = df_.dropna()
print(df_.shape)
df_.head(5)

# +
#df_[df_['location'] == 'Birx'].iloc[1604]['tweet_text']

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
show_tweets_locations(df_tweets, class_label='caution_and_advice' , locations=['Yosemite National Park', 'California'])

plot_disaster_mentions(df_loc_graph, 'caution_and_advice', 'Caution And Advice')

show_tweets_locations(df_tweets, class_label='caution_and_advice',
                      locations=loc20[:2], sample=None)

plot_disaster_mentions(df_loc_graph, 'displaced_people_and_evacuations', 'Displaced People And Evacuations')

show_tweets_locations(df_tweets, class_label='displaced_people_and_evacuations',
                      locations=loc20[:2])

plot_disaster_mentions(df_loc_graph, 'injured_or_dead_people', 'Injured Or Dead People')

show_tweets_locations(df_tweets, class_label='injured_or_dead_people',
                      locations=loc20[:2], sample=10)

plot_disaster_mentions(df_loc_graph, 'rescue_volunteering_or_donation_effort',
                       'Rescue Volunteering Or Donation Effort')

show_tweets_locations(df_tweets, class_label='rescue_volunteering_or_donation_effort',
                      locations=loc20[:2], sample=10)

plot_disaster_mentions(df_loc_graph, 'requests_or_urgent_needs',
                       'requests_or_urgent_needs')

show_tweets_locations(df_tweets, class_label='requests_or_urgent_needs',
                      locations=loc20[:2], sample=10)

plot_disaster_mentions(df_loc_graph, 'sympathy_and_support',
                       'sympathy_and_support')

show_tweets_locations(df_tweets, class_label='sympathy_and_support',
                      locations=loc20[:2], sample=10)

plot_disaster_mentions(df_loc_graph, 'infrastructure_and_utility_damage',
                       'infrastructure_and_utility_damage')

show_tweets_locations(df_tweets, class_label='infrastructure_and_utility_damage',
                      locations=loc20[:2], sample=10)


