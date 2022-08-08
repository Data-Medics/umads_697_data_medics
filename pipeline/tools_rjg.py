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
            df = pd.read_csv(f'../data/HumAID_data_v1.0/event_type/{disaster}_{dataset}.tsv', sep='\t')
            df['disaster_type'] = disaster
            combined_df = pd.concat([combined_df, df])
    return combined_df


def tweet_preprocessing(tweet_string: str, min_word_length: int = 3):
    """function to preprocess the tweet text string"""
    # remove twitter handles (@user)
    text = re.sub(r'@[\S]*', '', tweet_string)
    regex = re.compile(r'[^a-zA-Z]')
    # remove string that does not start with a letter
    text = regex.sub(' ', text)
    # remove web address
    text = re.sub(r'http[^\\S]*', '', text)
    # remove numbers
    text = text.replace(r'[0-9]', " ")
    # remove hashtag
    text = re.sub(r'#[\S]*', '', text)
    # # split compound words
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    # remove non-ascii
    text = text.replace(r'[^\x00-\x7F]+', " ")
    # remove short words
    text = ' '.join([w for w in text.split() if len(w) > min_word_length])
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
    tokenized.append(' '.join([token.lemma_ if token.lemma_ not in ['-PRON-']
                                               and token.pos_ not in ['PROPN'] else ''
                                                for token in doc if token.pos_ in allowed_postags]))
    return ''.join(tokenized)


def get_dominant_topics(fitted_lda_model, vectorized_text):
    """ extract the dominant topics from each set of tweets"""
    lda_output = fitted_lda_model.transform(vectorized_text)
    topicnames = ["Topic" + str(i) for i in range(fitted_lda_model.n_components)]
    # index names
    docnames = ["Doc" + str(i) for i in range(vectorized_text.toarray().shape[0])]
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


def get_topic_distribution(fitted_lda_model, fitted_vectorizer):
    """get topic distribution"""
    # Get topic names
    topicnames = ["Topic" + str(i) for i in range(fitted_lda_model.n_components)]

    # Topic-Keyword Matrix
    df_topic_keywords = pd.DataFrame(fitted_lda_model.components_)

    # Assign Column and Index
    df_topic_keywords.columns = fitted_vectorizer.get_feature_names()
    df_topic_keywords.index = topicnames

    # View
    return df_topic_keywords


def show_topics(fitted_vectorizer, fitted_lda_model, n_words=100, dname=''):
    """Show top n keywords for each topic"""
    keywords = np.array(fitted_vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in fitted_lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    df_topic_keywords = pd.DataFrame(topic_keywords)
    df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
    df_topic_keywords.index = [str(dname)+'_Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
    return df_topic_keywords
    return df_topic_keywords


def extract_most_representative_sentence(fitted_lda_model, vectorized_text, filtered_disaster_df):
    """extract the most representative sentence for each topic"""
    lda_output = fitted_lda_model.transform(vectorized_text)
    topicnames = [i for i in range(fitted_lda_model.n_components)]
    # index names
    docnames = [i for i in range(vectorized_text.toarray().shape[0])]
    # Make the pandas dataframe
    df_document_topic = pd.DataFrame(np.round(lda_output, 4), columns=topicnames, index=docnames)
    most_rep_sentence = {}
    for i in topicnames:
        id = df_document_topic[i].idxmax()
        most_rep_sentence['Topic_'+str(i)] = ([id, filtered_disaster_df.iloc[[id]]])

    return most_rep_sentence
