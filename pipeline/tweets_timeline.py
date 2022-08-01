# # Doing an exploration on recent disaster tweets

# +
import pandas as pd
import pickle

import spacy
from spacy import displacy

from tools import extract_assign_location

# + tags=["parameters"]
upstream = ['category_classification_models',
            'recent_tweets_wildfire']
random_seed = 42
# -

# ## Load the sample of recent tweets



# ## Extract the location from the tweet sample

# + active=""
# df_disaster_tweets, df_locs_most_common = extract_assign_location(df_disaster_tweets, number_most_common=10)
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
