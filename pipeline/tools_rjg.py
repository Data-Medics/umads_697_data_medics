import numpy as np
import pandas as pd
import re, nltk, spacy, gensim
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def generate_disaster_type_dataframe(disaster_types: tuple = ('earthquake', 'fire', 'flood', 'hurricane'),
                                     dev_train_test: tuple = ('dev', 'train', 'test')):
    """function to gather all the disaster tweets, combine them and label them accordingly"""
    combined_df = pd.DataFrame()
    disaster_dict = {}
    for disaster in disaster_types:
        for dataset in dev_train_test:
            df = pd.read_csv(f'../data/HumAID_data_v1.0/event_type/{disaster}_{dataset}.tsv')
            df['disaster_type'] = disaster
            pd.concat([combined_df, df])
    return combined_df


def tweet_preprocessing(tweet_string: str, min_word_length: int = 3):
    """function to preprocess the tweet text string"""
    # remove twitter handles (@user)
    text = re.sub(r'@[\S]*', '', x)
    regex = re.compile(r'[^a-zA-Z]')
    # remove string that does not start with a letter
    text = regex.sub(' ', text)
    # remove web address
    text = re.sub(r'http[^\\S]*', '', text)
    # remove numbers
    text = text.replace(r'[0-9]', " ")
    # remove hashtag
    text = re.sub(r'#[\S]*', '', text)
    # split compound words
    text = re.sub(r'([A-Z])', r' \1', text)
    # remove non-ascii
    text = text.replace(r'[^\x00-\x7F]+', " ")
    # remove short words
    text = ' '.join([w for w in text.split() if len(w) >= min_word_length])
    # lowercase
    text = text.lower()
    # tokenize
    text = word_tokenize(text)
    return text


def lemmatize_tweet_text(tweet_text: str, nlp=spacy.load('en_core_web_sm', disable=['parser', 'ner']),
                         allowed_postags: tuple = ('NOUN', 'ADJ', 'VERB', 'ADV')):
    """https://spacy.io/api/annotation"""
    tokenized = []
    doc = nlp(' '.join(tweet_text))
    # exclude pronouns and tokenize
    tokenized.append(' '.join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token
                               in doc if token.pos_ in allowed_postags]))
    return ''.join(tokenized)


def get_dominant_topics(fitted_lda_model, vectorized_text):
    """ extract the dominant topics from each set of tweets"""
    lda_output = fitted_lda_model.transform(vectorized_text)
    topicnames = ["Topic" + str(i) for i in range(fitted_lda_model.n_components)]
    # index names
    docnames = ["Doc" + str(i) for i in range(vectorized_text.toarray().shape()[0])]
    # Make the pandas dataframe
    df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)
    # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic'] = dominant_topic

    # Styling
    def color_green(val):
        color = 'green' if val > .1 else 'black'
        return 'color: {col}'.format(col=color)

    def make_bold(val):
        weight = 700 if val > .1 else 400
        return 'font-weight: {weight}'.format(weight=weight)

    # Apply Style
    df_document_topics = df_document_topic.head(15).style.applymap(color_green).applymap(make_bold)
    return df_document_topics
