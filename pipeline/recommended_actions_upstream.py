import yaml

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
limit = 5000
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

df = query_tweets(bearer_token=bearer_token, query=query + ' lang:' + language, limit=limit)

# hash the tweet text to count unique tweets
df["tweet_hash"] = df["tweet_text"].apply(lambda a: hash(a) % 100000000)

# counts by tweet
df["tweet_count"] = df.groupby("tweet_hash").transform("count")["tweet_text"]

df_slim = df.groupby("tweet_hash").first()

# df_slim.to_csv(product['file'], index=False)
df_slim.to_csv("output/twitter_actions.csv", index=False)


# -


