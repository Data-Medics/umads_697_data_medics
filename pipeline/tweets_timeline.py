# # Doing an exploration on recent disaster tweets for a disaster category

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
import warnings

from tools import extract_assign_location, disaster_title, topNlocations, top_locations, \
                    sampleNtweets, top_locations_tweets, show_top_locations, get_top_locations_flatten, \
                    get_location_graph_data, build_network, get_layout, plot_disaster_mentions, is_intersect, \
                    show_tweets_locations, top_location_pairs

warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', 280)
# -

# ## Define any upstream dependencies

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

print('Disaster kind: ', disaster_title(disaster_kind))

# ## Load the sample of recent tweets

df_tweets = pd.read_csv(upstream[disaster_kind]['file'])
df_tweets.sample(5)

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

# ### Prepare the time series grouped by disaster class label 

# +
df_tweets['created_at'] = pd.to_datetime(df_tweets['created_at'], errors='coerce')
df_plot = df_tweets.copy()

df_plot['created_at'] = df_plot['created_at'].dt.floor('8H') # Round to 8 hours
df_plot['count'] = 1
df_group_plot = df_plot[['created_at', 'class_label', 'count']].groupby(['created_at', 'class_label']).count().reset_index()
df_group_plot['created_at'] = pd.to_datetime(df_group_plot['created_at'])
df_group_plot.head()
# -

# ### Prepare the time series grouped by location label 

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

# ### Find the top 10 locations in the tweets sample

top10_locs = list(df_group_explode['location'].unique())
top10_locs

# +
chart = alt.Chart(df_group_explode[df_group_explode['location'].isin(top10_locs)]).mark_line(size=2).encode(
    x='created_at:T',
    y='count:Q',
    color='location:N',
    tooltip=['location', 'count']
)

chart.properties(
    width=500,
    height=500
).properties(
    title=disaster_title(disaster_kind) + ' disasters in the world, interactive chart'
).interactive()
# -

# ## Combine the most important locations per category

df_top_locations = top_locations(df_tweets, top10_locs)
df_top_locations.sample(5)

# ### Define the class labels we are interested in

class_labels = [
    'displaced_people_and_evacuations',
    'injured_or_dead_people',
    'rescue_volunteering_or_donation_effort',
    'requests_or_urgent_needs',
    'sympathy_and_support',
    'infrastructure_and_utility_damage',
    'caution_and_advice'
]

# +
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
    title=disaster_title(disaster_kind) + ' disasters in the world, interactive chart'
)
# -

# ## Show sample tweets per most important locations above

df_top_location_tweets = top_locations_tweets(df_tweets, top10_locs)
df_top_location_tweets.sample(5)

show_top_locations(df_top_location_tweets, class_labels, disaster_kind)

# ## Visualize the locations and relationship between them as a network

# ### Select the top 20 locations for this disaster

loc20 = get_top_locations_flatten(df_tweets, 20)
loc20

# ### Get the tweets for just the top 20 locations

df_loc_graph = get_location_graph_data(df_tweets, loc20)
df_loc_graph.sample(10)

plot_disaster_mentions(df_loc_graph, 'caution_and_advice', 'Caution And Advice')

# ### Get the most frequent location pair

loc_pairs = top_location_pairs(df_loc_graph, 'caution_and_advice', top_pairs_count=1)
loc_pairs

# ### Show a sample for this location pair

show_tweets_locations(df_tweets, class_label='caution_and_advice' , locations=loc_pairs[0], sample=5)

plot_disaster_mentions(df_loc_graph, 'displaced_people_and_evacuations', 'Displaced People And Evacuations')

# ### Get the most frequent location pair

loc_pairs = top_location_pairs(df_loc_graph, 'displaced_people_and_evacuations', top_pairs_count=1)
loc_pairs

# ### Show a sample for this location pair

show_tweets_locations(df_tweets, class_label='displaced_people_and_evacuations' , locations=loc_pairs[0], sample=5)

plot_disaster_mentions(df_loc_graph, 'injured_or_dead_people', 'Injured Or Dead People')

# ### Get the most frequent location pair

loc_pairs = top_location_pairs(df_loc_graph, 'injured_or_dead_people', top_pairs_count=1)
loc_pairs

# ### Show a sample for this location pair

show_tweets_locations(df_tweets, class_label='injured_or_dead_people' , locations=loc_pairs[0], sample=5)

plot_disaster_mentions(df_loc_graph, 'rescue_volunteering_or_donation_effort',
                       'Rescue Volunteering Or Donation Effort')

# ### Get the most frequent location pair

loc_pairs = top_location_pairs(df_loc_graph, 'rescue_volunteering_or_donation_effort', top_pairs_count=1)
loc_pairs

# ### Show a sample for this location pair

show_tweets_locations(df_tweets, class_label='rescue_volunteering_or_donation_effort',
                      locations=loc_pairs[0], sample=5)

plot_disaster_mentions(df_loc_graph, 'requests_or_urgent_needs',
                       'requests_or_urgent_needs')

# ### Get the most frequent location pair

loc_pairs = top_location_pairs(df_loc_graph, 'requests_or_urgent_needs', top_pairs_count=1)
loc_pairs

# ### Show a sample for this location pair

show_tweets_locations(df_tweets, class_label='requests_or_urgent_needs',
                      locations=loc_pairs[0], sample=5)

plot_disaster_mentions(df_loc_graph, 'sympathy_and_support',
                       'sympathy_and_support')

# ### Get the most frequent location pair

loc_pairs = top_location_pairs(df_loc_graph, 'sympathy_and_support', top_pairs_count=1)
loc_pairs

# ### Show a sample for this location pair

show_tweets_locations(df_tweets, class_label='sympathy_and_support',
                      locations=loc_pairs[0], sample=5)

plot_disaster_mentions(df_loc_graph, 'infrastructure_and_utility_damage',
                       'infrastructure_and_utility_damage')

# ### Get the most frequent location pair

loc_pairs = top_location_pairs(df_loc_graph, 'infrastructure_and_utility_damage', top_pairs_count=1)
loc_pairs

# ### Show a sample for this location pair

show_tweets_locations(df_tweets, class_label='infrastructure_and_utility_damage',
                      locations=loc_pairs[0], sample=5)


