# + tags=[]
import os
import pandas as pd
import pickle
import sys
import spacy
from spacy.language import Language
import re
# -

from sklearn.linear_model import LogisticRegression

# + tags=["parameters"]
# once we have tweets of interest the upstream will change
# to the data generation step we are actually interested in
upstream = ["recommended_actions_upstream", "train_logistic_regression"]

# +
# load a spacy language model
nlp = spacy.load("en_core_web_sm")
stopwords = nlp.Defaults.stop_words
# -


# df = pd.read_csv(params['file'])
df = pd.read_csv("output/twitter_actions.csv")

# load the vectorizer
vectorizer = pickle.load(open(os.path.join(".", "output", "fitted_models", "vectorizer.pkl"), "rb"))

# load the model
clf_model = pickle.load(open(os.path.join(".", "output", "fitted_models", "lr_model.pkl"), "rb"))

# prepare text for model - vectorize the tweets 
raw_tweets_vectorized = vectorizer.transform(df['tweet_text'])

tweet_class_preds = clf_model.predict(raw_tweets_vectorized)

df["predicted_class"] = tweet_class_preds

# filter to only the tweets we are interested in - those callling for an action
action_tweets = df[df.predicted_class == "rescue_volunteering_or_donation_effort"].copy()
action_tweets = action_tweets.sort_values("tweet_count", ascending=False)

action_tweets["spacy_text"] = action_tweets["tweet_text"].apply(nlp)


def make_str(list_of_verbs):
    list_of_verbs = [a.lower() for a in list_of_verbs]
    if len(list_of_verbs) == 1:
        return list_of_verbs[0]
    else:
        return ' or '.join(set(list_of_verbs))


verb_list = ["donate", "volunteer", "evacuate"]
regex = re.compile('|'.join(re.escape(x) for x in verb_list), re.IGNORECASE)

for idx, data in action_tweets.iterrows():
    # find the recommended action
    verb_matches = re.findall(regex, data["tweet_text"])
    total_tweet_count = data.tweet_count - 1
    
    # at least on word has been found
    if len(verb_matches) > 0:
        
        # find all the links (often more than 1 donation site)
        donation_url_list = []
        
        # check for a retweet
        original_tweeter = re.findall("RT @([a-zA-z0-9_]*)", data["tweet_text"])
        
        # find and record all the urls in the tweet
        for token in data["spacy_text"]:
            if token.like_url:
                donation_url_list.append(token)

        
        if len(donation_url_list) > 0:
            print(f"Original tweet:\n{data['tweet_text']}\n")
            if len(original_tweeter) > 0:
                tweet_author = original_tweeter[0]
            else:
                tweet_author = data["name"]
            for idx, url in enumerate(donation_url_list):
                if idx == 0:
                    print(f"{tweet_author} and {total_tweet_count} others recommend you {make_str(verb_matches)}.  More information at {url}")
                else:
                    print(f"Please also consider donating to {url}")
            print("\n\n")