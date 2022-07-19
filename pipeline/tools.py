import pandas as pd
import numpy as np
import re

from collections import Counter


def get_locations(nlp, df, limit_most_common=100):
    locations = []

    for _, row in df.iterrows():
        doc = nlp(row['tweet_text'])
        locations.extend([[ent.text] for ent in doc.ents if ent.label_ in ['LOC']])

    df_locs = pd.DataFrame(locations, columns=['Location'])

    locations = Counter(df_locs['Location']).most_common(limit_most_common)
    df_locs_most_common = pd.DataFrame(locations, columns=['location', 'count'])
    locations_set = {l.lower() for l in df_locs_most_common['location'] if not l.startswith('@')}

    return df_locs, locations_set


class Tokenizer:
    def __init__(self, nlp, stopwords, stemmer=None):
        self.nlp = nlp
        self.stopwords = stopwords
        self.stemmer = stemmer
        self.nlp.Defaults.stop_words |= stopwords

    def tokenize(self, text):
        # Tokenize
        doc = self.nlp(text.lower())

        # Lematize, stop words and lematization removal
        tokens = [token.lemma_ for token in doc if not (token.is_stop or token.is_punct or token.is_digit)]

        # Stemming - uncomment to try it out
        if self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]

        # Check more here - noun chunks, entities etc - https://stackabuse.com/python-for-nlp-tokenization-stemming-and-lemmatization-with-spacy-library/

        # Return the tokens
        return tokens
