import yaml
import datetime
import pandas as pd

from tools_zupan import get_hashtags, query_tweets

# + tags=["parameters"]
upstream = []
disaster_types = None
action_types = None
# query=None
language=None
limit=None
credentials_file=None

query = '(wildfire OR flood OR earthquake donate evacuate volunteer)'
disaster_types = ["wildfire", "flood", "earthquake"]
action_types = ["volunteer", "donate", "evacuate"]
language = 'en'
limit = 500
credentials_file = "credentials.yaml"

# +
queries = []
for d in disaster_types:
    for a in action_types:
        queries.append(d + " "+ a)
        
query = "(" + " OR ".join(queries) + ")" 

# +
with open(credentials_file, 'r') as stream:
    credentials = yaml.safe_load(stream)

bearer_token = credentials['twitter_bearer_token']

# df = query_tweets(bearer_token=bearer_token, query=query + ' lang:' + language, limit=limit)

skip_interval_hours = 6
periods_one_week = 24
period_end = datetime.datetime.now()
period_delta = datetime.timedelta(hours=skip_interval_hours)

full_df = None

for _ in range(periods_one_week):
    df = query_tweets(bearer_token=bearer_token, query=query + ' lang:' + language, 
                      limit=limit,  end_time=period_end)
    if full_df is None:
        full_df = df
    else:
        full_df = pd.concat([full_df, df])
    period_end -= period_delta


# +
# skip_interval_hours = 6
# periods_one_week = 24
# period_end = datetime.datetime.now()
# period_delta = datetime.timedelta(hours=skip_interval_hours)

# full_df = None

# for _ in range(periods_one_week):
#     print(period_end)
#     period_end -= period_delta

# +
# hash the tweet text to count unique tweets
full_df["tweet_hash"] = full_df["tweet_text"].apply(lambda a: hash(a) % 100000000)

# counts by tweet
full_df["tweet_count"] = full_df.groupby("tweet_hash").transform("count")["tweet_text"]

df_slim = full_df.groupby("tweet_hash").first()

# df_slim.to_csv(product['file'], index=False)
print(df_slim.shape)
df_slim.to_csv("output/twitter_actions.csv", index=False)
# -


