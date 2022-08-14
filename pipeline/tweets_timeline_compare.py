# # Doing an exploration on recent disaster tweets for all disaster types

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

from tools import extract_assign_location_per_kind, disaster_title, topNlocations, top_locations, \
                    sampleNtweets, top_locations_tweets, show_top_locations, get_top_locations_flatten, \
                    get_location_graph_data, build_network, get_layout, plot_disaster_mentions, is_intersect, \
                    show_tweets_locations, top_location_pairs, plot_class_boxplot

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
disaster_kinds = None


# -

disaster_kind_names = [disaster_title(disaster_kind) for disaster_kind in disaster_kinds]
disaster_kind_names

# ## Load the sample of recent tweets for all disaster types

# +
df_tweets = None

for disaster_kind in disaster_kinds:
    _df = pd.read_csv(upstream[disaster_kind]['file'])
    _df['disaster_kind'] = disaster_title(disaster_kind)
    
    if df_tweets is None:
        df_tweets = _df
    else:
        df_tweets = df_tweets.append(_df, ignore_index=True)

print('Number of tweets: ', df_tweets.shape[0])
df_tweets.sample(5)
# -

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

# ## Plot the intensity for each disaster kind, each category

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
df_tweets['created_at'] = pd.to_datetime(df_tweets['created_at'], errors='coerce')
df_plot = df_tweets.copy()

df_plot['created_at'] = df_plot['created_at'].dt.floor('8H') # Round to 8 hours
df_plot['count'] = 1
df_group_plot = df_plot[['created_at', 'class_label', 'disaster_kind','count']].groupby(
    ['created_at', 'class_label', 'disaster_kind']
).count().reset_index()
df_group_plot['created_at'] = pd.to_datetime(df_group_plot['created_at'])
df_group_plot.sort_values(by='disaster_kind', inplace=True)
df_group_plot.head()
# -

# ### Looking at all the disaters at once does not have any indication which one dominates

plot_class_boxplot(df_group_plot)

plot_class_boxplot(df_group_plot, 'displaced_people_and_evacuations')

plot_class_boxplot(df_group_plot, 'injured_or_dead_people')

plot_class_boxplot(df_group_plot, 'rescue_volunteering_or_donation_effort')

plot_class_boxplot(df_group_plot, 'requests_or_urgent_needs')

plot_class_boxplot(df_group_plot, 'sympathy_and_support')

plot_class_boxplot(df_group_plot, 'infrastructure_and_utility_damage')

plot_class_boxplot(df_group_plot, 'caution_and_advice')

plot_class_boxplot(df_group_plot, 'not_humanitarian')

plot_class_boxplot(df_group_plot, 'other_relevant_information')

class_labels_active = [
    'displaced_people_and_evacuations',
    'injured_or_dead_people',
]

plot_class_boxplot(df_group_plot[df_group_plot['class_label'].isin(class_labels_active)], title=class_labels_active)

class_labels_past = [
    'rescue_volunteering_or_donation_effort',
    'requests_or_urgent_needs',
    'sympathy_and_support',
]

plot_class_boxplot(df_group_plot[df_group_plot['class_label'].isin(class_labels_past)], title=class_labels_past)

plt.figure(figsize = (10,6))
sns.lineplot(x='created_at', y='count', hue="disaster_kind", style="disaster_kind", ci=None,
             data=df_group_plot[df_group_plot['class_label'].isin(class_labels_active)]);

plt.figure(figsize = (10,6))
sns.lineplot(x='created_at', y='count', hue="disaster_kind", style="disaster_kind", ci=None,
             data=df_group_plot[df_group_plot['class_label'].isin(class_labels_past)]);



df_group_plot[df_group_plot['class_label'].isin(class_labels_active)].groupby('disaster_kind').agg(
    {
        "count": ['min', 'max', 'mean', 'std']
    }
)

df_group_plot[df_group_plot['class_label'].isin(class_labels_past)].groupby('disaster_kind').agg(
    {
        "count": ['min', 'max', 'mean', 'std']
    }
)


