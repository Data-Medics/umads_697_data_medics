

import os
import pandas as pd
import pickle
import spacy
import re
from sklearn.linear_model import LogisticRegression
import itertools

import locations as loc

data_path = os.path.join(loc.data, "all_combined", "all_train.tsv")
training_data_sample = pd.read_csv(data_path, nrows=5000, sep="\t")
training_data_sample.to_csv(os.path.join(loc.blog_data, "train_data_sample.csv"), index=False)


def get_locs(a):
    locs = [ent.text for ent in a.ents if ent.label_ in ['LOC', 'GPE']]
    locs.append('all')
    return locs


# +
nlp = spacy.load("en_core_web_sm")
stopwords = nlp.Defaults.stop_words

# df = pd.read_csv(params['file'])
df = pd.read_csv("output/twitter_actions.csv")

# load the vectorizer
vectorizer = pickle.load(open(os.path.join("output", "vectorizer.pkl"), "rb"))

# load the model
clf_model = pickle.load(open(os.path.join("output", "model_lr.pkl"), "rb"))

# prepare text for model - vectorize the tweets 
raw_tweets_vectorized = vectorizer.transform(df['tweet_text'])

tweet_class_preds = clf_model.predict(raw_tweets_vectorized)

df["predicted_class"] = tweet_class_preds

# add spacy nlp
df["spacy_text"] = df["tweet_text"].apply(nlp)

df["locations"] = df["spacy_text"].apply(lambda a: get_locs(a))
# -

df.to_csv(os.path.join(loc.blog_data, "classification_data_sample.csv"), index=False)

locations_list = list(set(itertools.chain.from_iterable([a for a in df["locations"].values if len(a) > 0])))

with open(os.path.join(loc.blog_data, "locations"), "wb") as l:
    pickle.dump(locations_list, l)

# filter to only the tweets we are interested in - those callling for an action
action_tweets = df[df.predicted_class == "rescue_volunteering_or_donation_effort"].copy()

action_tweets.to_csv(os.path.join(loc.blog_data, "action_tweets_data_sample.csv"), index=False)


