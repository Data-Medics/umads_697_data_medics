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

def space(num_lines=1):
    """Adds empty lines to the Streamlit app."""
    for _ in range(num_lines):
        st.write("")

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

	# add spacy nlp
	df["spacy_text"] = df["tweet_text"].apply(nlp)

	df["locations"] = df["spacy_text"].apply(lambda a: [ent.text for ent in a.ents if ent.label_ in ['LOC']])

	return df



st.title('Disaster Tweets By Category')

space(1)

tweet_categories = ['rescue_volunteering_or_donation_effort',
       'other_relevant_information', 'requests_or_urgent_needs',
       'injured_or_dead_people', 'infrastructure_and_utility_damage',
       'sympathy_and_support', 'caution_and_advice', 'not_humanitarian',
       'displaced_people_and_evacuations', 'missing_or_found_people']

tweet_types = st.multiselect("Choose a tweet topic", tweet_categories, tweet_categories[:])

start_date = st.date_input(
    "Select Minimum Tweet Date",
    datetime.date(2020, 1, 1),
    min_value=datetime.datetime.strptime("2020-01-01", "%Y-%m-%d"),
    max_value=datetime.datetime.now(),
)
print(type(start_date))
tweet_data = get_tweet_class_prediction()

date_filer = pd.to_datetime(tweet_data.created_at).apply(lambda x: x.date()) >= start_date

tweet_data_filtered = tweet_data[(tweet_data.predicted_class.isin(tweet_types)) & (date_filer)]

tweet_data_filtered["created_at"] = pd.to_datetime(tweet_data_filtered.created_at).apply(lambda x: x.date())

locations = list(itertools.chain.from_iterable([a for a in tweet_data_filtered["locations"].values if len(a) > 0]))

location_list = st.multiselect("Specify a location", locations, locations[:])

weet_data_filtered = tweet_data_filtered[tweet_data_filtered.locations.isin(location_list)]

tweet_data_filtered = tweet_data_filtered[["created_at", "tweet_text", "name", "tweet_count"]].copy()


st.write(st.table(tweet_data_filtered))




