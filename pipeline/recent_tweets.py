# # Retreve tweets from the past one week by query

# +
import pandas as pd
import yaml

from tools import retrieve_tweets

# + tags=["parameters"]
upstream = []
product = None
disaster_type = None
query = None
language = None
limit = None
skip_interval_hours=None
credentials_file = None
# -

# ## Load Twitter credentials

# +
with open(credentials_file, 'r') as stream:
    credentials = yaml.safe_load(stream)

bearer_token = credentials['twitter_bearer_token']
# -

# ## Sample data from Twitter

# %%time
df = retrieve_tweets(bearer_token, skip_interval_hours, query, language, limit)
df.sample(10)

df.to_csv(product['file'], index=False)


