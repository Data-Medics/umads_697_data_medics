# + tags=[]
import os
import pandas as pd
import sys
import spacy
from spacy.language import Language
import tweepy
import re

# + tags=["parameters"]
# once we have tweets of interest the upstream will change
# to the data generation step we are actually interested in
upstream = ["recommended_actions_upstream"]

# + tags=["injected-parameters"]
# load a spacy language model
nlp = spacy.load("en_core_web_sm")
stopwords = nlp.Defaults.stop_words
# -


# df = pd.read_csv(params['file'])
df = pd.read_csv("output/x.csv")

df.head()

df["spacy_text"] = df["tweet_text"].apply(nlp)


def make_str(list_of_verbs):
    list_of_verbs = [a.lower() for a in list_of_verbs]
    if len(list_of_verbs) == 1:
        return list_of_verbs[0]
    else:
        return ' or '.join(set(list_of_verbs))


verb_list = ["donate", "volunteer", "evacuate"]
regex = re.compile('|'.join(re.escape(x) for x in verb_list), re.IGNORECASE)

for idx, data in df.iterrows():
    # find the recommended action
    verb_matches = re.findall(regex, data["tweet_text"])
    
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
            for url in donation_url_list:
                print(f"{tweet_author} recommends you {make_str(verb_matches)}.  More information at {url}\n")
            print("\n\n")
