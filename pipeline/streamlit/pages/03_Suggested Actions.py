import streamlit as st
import pandas as pd
import datetime
import os
import pandas as pd
import pickle
import sys
import spacy
from spacy.language import Language
import re
from sklearn.linear_model import LogisticRegression
import itertools
from typing import List

def space(num_lines=1):
    """Adds empty lines to the Streamlit app."""
    for _ in range(num_lines):
        st.write("")

def make_str(list_of_verbs: List[str]) -> str:
    list_of_verbs = [a.lower() for a in list_of_verbs]
    if len(list_of_verbs) == 1:
        return list_of_verbs[0]
    else:
        return ' or '.join(set(list_of_verbs))

def get_tweet_class_prediction():
    # load a spacy language model
    nlp = spacy.load("en_core_web_sm")
    stopwords = nlp.Defaults.stop_words

    # df = pd.read_csv(params['file'])
    df = pd.read_csv("../output/twitter_actions.csv")

    # load the vectorizer
    vectorizer = pickle.load(open(os.path.join("..", "output", "fitted_models", "vectorizer.pkl"), "rb"))

    # load the model
    clf_model = pickle.load(open(os.path.join("..", "output", "fitted_models", "lr_model.pkl"), "rb"))

    # prepare text for model - vectorize the tweets 
    raw_tweets_vectorized = vectorizer.transform(df['tweet_text'])

    tweet_class_preds = clf_model.predict(raw_tweets_vectorized)

    df["predicted_class"] = tweet_class_preds

    # filter to only the tweets we are interested in - those callling for an action
    action_tweets = df[df.predicted_class == "rescue_volunteering_or_donation_effort"].copy()

    # add spacy nlp
    action_tweets["spacy_text"] = action_tweets["tweet_text"].apply(nlp)

    return action_tweets



st.title('Recommended Actions')

space(1)

col1, col2 = st.columns(2)

with col1:
    start_date = st.date_input(
        "Select Minimum Tweet Date",
        datetime.date(2020, 1, 1),
        min_value=datetime.datetime.strptime("2020-01-01", "%Y-%m-%d"),
        max_value=datetime.datetime.now(),
    )

with col2:
    verb_list = ["donate", "volunteer", "evacuate"]

    verb_types = st.multiselect("Choose a tweet topic", verb_list, verb_list[:])


tweet_data = get_tweet_class_prediction()

date_filter = pd.to_datetime(tweet_data.created_at).apply(lambda x: x.date()) >= start_date

tweet_data_filtered = tweet_data[date_filter].copy()

space(2)


regex = re.compile('|'.join(re.escape(x) for x in verb_list), re.IGNORECASE)

for idx, data in tweet_data_filtered.iterrows():
    # find the recommended action
    verb_matches = re.findall(regex, data["tweet_text"])
    if len(set(verb_matches).intersection(set(verb_types))) < 1:
        continue
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
            if len(original_tweeter) > 0:
                tweet_author = original_tweeter[0]
            else:
                tweet_author = data["name"]
            for idx, url in enumerate(donation_url_list):
                if idx == 0:
                    st.write(f"{tweet_author} and {total_tweet_count} others recommend you {make_str(verb_matches)}.  More information at {url}")
                else:
                    st.write(f"Please also consider donating to {url}")
            with st.expander("Original Tweet"):
                st.write(data['tweet_text'])
            st.write("\n\n")
            st.write("-"*50)




