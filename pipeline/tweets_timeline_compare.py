# # Doing an exploration on recent disaster tweets for all disaster types - determine if one disaster dominates in certain class/category

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
import scipy.stats as stats

from tools import extract_assign_location_per_kind, disaster_title, topNlocations, top_locations, \
                    sampleNtweets, top_locations_tweets, show_top_locations, get_top_locations_flatten, \
                    get_location_graph_data, build_network, get_layout, plot_disaster_mentions, is_intersect, \
                    show_tweets_locations, top_location_pairs, plot_class_boxplot, f_test, ks_test

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

# ## Predict the class labels - add class_label to all the tweets 

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

# ## Plot the intensity for each disaster kind, each class

class_labels = [
    'displaced_people_and_evacuations',
    'injured_or_dead_people',
    'rescue_volunteering_or_donation_effort',
    'requests_or_urgent_needs',
    'sympathy_and_support',
    'infrastructure_and_utility_damage',
    'caution_and_advice'
]

# ### Aggregate the disaster types and categories in 8-hour groups

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

# ### Looking at all the disaters at once does not have any indication which one dominates. Visually, when no class is considered, noone dominates.

plot_class_boxplot(df_group_plot)

# ### Perform an f-test of the above, to determine if the means are different - the f-statistic is low and the pvalue is high, which confirms that no disaster kind stands up as compared to the others

f_test(df_group_plot)

# ### The outcome is the same when we try a KS-test - no one disaster dominates the tweet sample

ks_test(df_group_plot)

# ### Trying displaced_people_and_evacuations - visually 'Wildfires' tweets are more dominant

plot_class_boxplot(df_group_plot, 'displaced_people_and_evacuations')

# ### The f-statistic is relatively low - below 10, but the pvalue is also low, so there is a chance at least one tweet distribution is different, however, not that statistically significant 

f_test(df_group_plot, class_labels=['displaced_people_and_evacuations'])

ks_test(df_group_plot, class_labels=['displaced_people_and_evacuations'])

plot_class_boxplot(df_group_plot, 'injured_or_dead_people')

f_test(df_group_plot, class_labels=['injured_or_dead_people'])

ks_test(df_group_plot, class_labels=['injured_or_dead_people'])

# ### For rescue_volunteering_or_donation_effort, the 'flood' disasters clearly dominate as shown visually and statistically 

plot_class_boxplot(df_group_plot, 'rescue_volunteering_or_donation_effort')

f_test(df_group_plot, class_labels=['rescue_volunteering_or_donation_effort'])

ks_test(df_group_plot, class_labels=['rescue_volunteering_or_donation_effort'])

# ### No clear dominant disaster kind for requests_or_urgent_needs

plot_class_boxplot(df_group_plot, 'requests_or_urgent_needs')

f_test(df_group_plot, class_labels=['requests_or_urgent_needs'])

ks_test(df_group_plot, class_labels=['requests_or_urgent_needs'])

plot_class_boxplot(df_group_plot, 'sympathy_and_support')

f_test(df_group_plot, class_labels=['sympathy_and_support'])

ks_test(df_group_plot, class_labels=['sympathy_and_support'])

plot_class_boxplot(df_group_plot, 'infrastructure_and_utility_damage')

f_test(df_group_plot, class_labels=['infrastructure_and_utility_damage'])

ks_test(df_group_plot, class_labels=['infrastructure_and_utility_damage'])

plot_class_boxplot(df_group_plot, 'caution_and_advice')

f_test(df_group_plot, class_labels=['caution_and_advice'])

ks_test(df_group_plot, class_labels=['caution_and_advice'])

plot_class_boxplot(df_group_plot, 'not_humanitarian')

f_test(df_group_plot, class_labels=['not_humanitarian'])

ks_test(df_group_plot, class_labels=['not_humanitarian'])

plot_class_boxplot(df_group_plot, 'other_relevant_information')

f_test(df_group_plot, class_labels=['other_relevant_information'])

ks_test(df_group_plot, class_labels=['other_relevant_information'])

# ### Identify class labels for active disasters - these are more distressed tweets

class_labels_active = [
    'displaced_people_and_evacuations',
    'injured_or_dead_people',
]

# ### The visualization and the statistic tests clearly shows the Wildfires dominate for this particular week

plot_class_boxplot(df_group_plot[df_group_plot['class_label'].isin(class_labels_active)], title=class_labels_active)

f_test(df_group_plot, class_labels=class_labels_active)

ks_test(df_group_plot, class_labels=class_labels_active)

# ### Identify class labels for past/calm down disasters - the danger has already passed, the community deals with the consequences of the event

class_labels_past = [
    'rescue_volunteering_or_donation_effort',
    'requests_or_urgent_needs',
    'sympathy_and_support',
]

# ### Visually and statistically 'flood' disasters dominate for this week

plot_class_boxplot(df_group_plot[df_group_plot['class_label'].isin(class_labels_past)], title=class_labels_past)

f_test(df_group_plot, class_labels=class_labels_past)

ks_test(df_group_plot, class_labels=class_labels_past)

# ### Show the disasters over time as timeseries

plt.figure(figsize = (10,6))
sns.lineplot(x='created_at', y='count', hue="disaster_kind", style="disaster_kind", ci=None,
             data=df_group_plot[df_group_plot['class_label'].isin(class_labels_active)]);

plt.figure(figsize = (10,6))
sns.lineplot(x='created_at', y='count', hue="disaster_kind", style="disaster_kind", ci=None,
             data=df_group_plot[df_group_plot['class_label'].isin(class_labels_past)]);

# ### A simple computation allows us to also compare the means and standard deviations for the disaster kinds - by inspecting these, we can see which one is higher, even we don't have a statistical proof  

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
