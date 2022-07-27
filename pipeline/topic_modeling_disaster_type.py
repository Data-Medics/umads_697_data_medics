import numpy as np
import pandas as pd
import re, nltk, spacy, gensim
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
nltk.download('punkt')
from gensim.corpora import Dictionary
from tqdm import tqdm
import os
from gensim.models import LdaModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import scikitplot as skplt
from sklearn.metrics import (precision_recall_curve,PrecisionRecallDisplay)
from sklearn.metrics import confusion_matrix
import pickle

from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

import matplotlib.pyplot as plt
# %matplotlib inline

random_seed = 10

#tags= ["parameters"]
upstream = []


# +
### Identify - what type of disasters are in a block of tweets
# Load the disaster files
# Earthquake
df_earthquake_dev = pd.read_csv('../data/HumAID_data_v1.0/event_type/earthquake_dev.tsv', sep='\t')
df_earthquake_train = pd.read_csv('../data/HumAID_data_v1.0/event_type/earthquake_train.tsv', sep='\t')
df_earthquake_test = pd.read_csv('../data/HumAID_data_v1.0/event_type/earthquake_test.tsv', sep='\t')

df_earthquake_dev['disaster_type'] = 'earthquake'
df_earthquake_train['disaster_type'] = 'earthquake'
df_earthquake_test['disaster_type'] = 'earthquake'

df_earthquake_all = pd.concat([df_earthquake_dev, df_earthquake_train, df_earthquake_test])

# Fire
df_fire_dev = pd.read_csv('../data/HumAID_data_v1.0/event_type/fire_dev.tsv', sep='\t')
df_fire_train = pd.read_csv('../data/HumAID_data_v1.0/event_type/fire_train.tsv', sep='\t')
df_fire_test = pd.read_csv('../data/HumAID_data_v1.0/event_type/fire_test.tsv', sep='\t')

df_fire_dev['disaster_type'] = 'fire'
df_fire_train['disaster_type'] = 'fire'
df_fire_test['disaster_type'] = 'fire'

df_fire_all = pd.concat([df_fire_dev, df_fire_train, df_fire_test])

# Flood
df_flood_dev = pd.read_csv('../data/HumAID_data_v1.0/event_type/flood_dev.tsv', sep='\t')
df_flood_train = pd.read_csv('../data/HumAID_data_v1.0/event_type/flood_train.tsv', sep='\t')
df_flood_test = pd.read_csv('../data/HumAID_data_v1.0/event_type/flood_test.tsv', sep='\t')

df_flood_dev['disaster_type'] = 'flood'
df_flood_train['disaster_type'] = 'flood'
df_flood_test['disaster_type'] = 'flood'

df_flood_all = pd.concat([df_flood_dev, df_flood_train, df_flood_test])

# Hurricane
df_hurricane_dev = pd.read_csv('../data/HumAID_data_v1.0/event_type/hurricane_dev.tsv', sep='\t')
df_hurricane_train = pd.read_csv('../data/HumAID_data_v1.0/event_type/hurricane_train.tsv', sep='\t')
df_hurricane_test = pd.read_csv('../data/HumAID_data_v1.0/event_type/hurricane_test.tsv', sep='\t')

df_hurricane_dev['disaster_type'] = 'hurricane'
df_hurricane_train['disaster_type'] = 'hurricane'
df_hurricane_test['disaster_type'] = 'hurricane'

df_hurricane_all = pd.concat([df_hurricane_dev, df_hurricane_train, df_hurricane_test])

df_disaster_dev = pd.concat([df_earthquake_dev, df_fire_dev, df_flood_dev, df_hurricane_dev])
df_disaster_train = pd.concat([df_earthquake_train, df_fire_train, df_flood_train, df_hurricane_train])
df_disaster_test = pd.concat([df_earthquake_test, df_fire_test, df_flood_test, df_hurricane_test])

df_all = pd.concat([df_disaster_dev, df_disaster_train, df_disaster_test])

# -

df_all = df_all.sample(35000)
df_all.head()



# +
#clean, tokenize text
# -

def text_pre_processing(x):
    # remove twitter handles (@user)
    text = re.sub(r'@[\S]*','',x)
    regex = re.compile('[^a-zA-Z]')
    text = regex.sub(' ', text)
    # remove web address
    text = re.sub(r'http[^\\S]*','',text)
#     # remove special characters, punctuations
#     text = text.replace(r'[^a-zA-Z#]', " ")
    # remove numbers
    text = text.replace(r'[0-9]', " ")
    # remove hashtag
    text = re.sub(r'#[\S]*','',text)
#    split compound words
    text = re.sub( r"([A-Z])", r" \1", text)
#     #remove non-ascii
    text = text.replace(r'[^\x00-\x7F]+', " ")
    #remove short words
    text = ' '.join([w for w in text.split() if len(w)>=3])
    #lowercase
    text = text.lower()
    #tokenize
    text = word_tokenize(text)
    return text
df_all['tweet_text_cleaned'] = df_all['tweet_text'].apply(lambda x: text_pre_processing(x))
df_all.head()


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
df_all.iloc[0]

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


# ## Lemmatizer

# +
def lemmatize(tweet, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    tokenized = []
    doc = nlp(' '.join(tweet))
    tokenized.append(' '.join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
#     tokenized.append(' '.join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc])) #if token.pos_ in allowed_postags]))
    return ''.join(tokenized)

# Do lemmatization keeping only Noun, Adj, Verb, Adverb
df_all['lemmatized'] = df_all['tweet_text_cleaned'].apply(lambda x: lemmatize(x, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']))
# -

src_file_path = os.path.dirname(os.path.abspath("__file__"))
filename_lemmatized = os.path.join(src_file_path, 'output\lemmatized_text.csv')
df_all.to_csv(filename_lemmatized)

#

df_all.head()

#



# ## Vectorize

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words |= {"attach", "ahead", "rt",'www','tinyurl','com', 'https', 'http', 
               '&amp','amp', 'rt', 'bit', 'ly', 'bitly', 'trump', 'byte', 'bytes', 'donald','emoji', }
# +
# Build Disaster Type LDA Model
disaster_types = ['earthquake','fire', 'flood', 'hurricane']



vectorizer_params = {'analyzer' : 'word',
                    'min_df' : 10,
                    'stop_words' : stop_words,
                    'lowercase' : False,
                    'ngram_range' : (1,2)}

earthquake_vectorizer = CountVectorizer(**vectorizer_params).fit(list(df_all[df_all['disaster_type']=='earthquake']['lemmatized']))
fire_vectorizer = CountVectorizer(**vectorizer_params).fit(list(df_all[df_all['disaster_type']=='fire']['lemmatized']))
flood_vectorizer = CountVectorizer(**vectorizer_params).fit(list(df_all[df_all['disaster_type']=='flood']['lemmatized']))
hurricane_vectorizer = CountVectorizer(**vectorizer_params).fit(list(df_all[df_all['disaster_type']=='hurricane']['lemmatized']))
# -


# ## LDA Model

# +
params_earthquake = {'n_components' : 3,
          'max_iter' : 10,
          'learning_method' : 'online',
          'random_state' : random_seed,
          'batch_size' : 128,
          'evaluate_every' : -1,
          'n_jobs' : -1,
          'learning_decay' : .5}

params_fire = {'n_components' : 3,
          'max_iter' : 10,
          'learning_method' : 'online',
          'random_state' : random_seed,
          'batch_size' : 128,
          'evaluate_every' : -1,
          'n_jobs' : -1,
          'learning_decay' : .5}

params_flood = {'n_components' : 3,
          'max_iter' : 10,
          'learning_method' : 'online',
          'random_state' : random_seed,
          'batch_size' : 128,
          'evaluate_every' : -1,
          'n_jobs' : -1,
          'learning_decay' : .5}

params_hurricane = {'n_components' : 3,
          'max_iter' : 10,
          'learning_method' : 'online',
          'random_state' : random_seed,
          'batch_size' : 128,
          'evaluate_every' : -1,
          'n_jobs' : -1,
          'learning_decay' : .5}

lda_model_earthquake = LatentDirichletAllocation(**params_earthquake)
lda_model_fire = LatentDirichletAllocation(**params_fire)
lda_model_flood = LatentDirichletAllocation(**params_flood)
lda_model_hurricane = LatentDirichletAllocation(**params_hurricane)

# -


earthquake_text_vectorized = earthquake_vectorizer.transform(list(df_all[df_all['disaster_type']=='earthquake']['lemmatized']))
fire_text_vectorized = fire_vectorizer.transform(list(df_all[df_all['disaster_type']=='fire']['lemmatized']))
flood_text_vectorized = flood_vectorizer.transform(list(df_all[df_all['disaster_type']=='flood']['lemmatized']))
hurricane_text_vectorized = hurricane_vectorizer.transform(list(df_all[df_all['disaster_type']=='hurricane']['lemmatized']))

lda_model_earthquake.fit_transform(earthquake_text_vectorized)
lda_model_fire.fit_transform(fire_text_vectorized)
lda_model_flood.fit_transform(flood_text_vectorized)
lda_model_hurricane.fit_transform(hurricane_text_vectorized)

# ## Performance Stats

# +
# Log Likelyhood: Higher the better
print("Log Likelihood Earthquake: ", lda_model_earthquake.score(earthquake_text_vectorized))
print("Log Likelihood Fire: ", lda_model_fire.score(fire_text_vectorized))
print("Log Likelihood Flood: ", lda_model_flood.score(flood_text_vectorized))
print("Log Likelihood Hurricane: ", lda_model_hurricane.score(hurricane_text_vectorized))

# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
print("Perplexity Earthquake: ", lda_model_earthquake.perplexity(earthquake_text_vectorized))
print("Perplexity Fire: ", lda_model_fire.perplexity(fire_text_vectorized))
print("Perplexity Flood: ", lda_model_flood.perplexity(flood_text_vectorized))
print("Perplexity Hurricane: ", lda_model_hurricane.perplexity(hurricane_text_vectorized))
# -

#



#



#



#





# ## Dominant Topics

# +
def get_dominant_topics(lda_model, disaster_type, vectorized_text):

    lda_output = lda_model.transform(vectorized_text)
    
    topicnames = ["Topic" + str(i) for i in range(lda_model.n_components)]

    # index names
    docnames = ["Doc" + str(i) for i in range(len(df_all[df_all['disaster_type']==disaster_type]))]

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

get_dominant_topics(lda_model_earthquake, 'earthquake', earthquake_text_vectorized)

get_dominant_topics(lda_model_fire, 'fire', fire_text_vectorized)

get_dominant_topics(lda_model_flood, 'flood', flood_text_vectorized)

get_dominant_topics(lda_model_hurricane, 'hurricane', hurricane_text_vectorized)


# -

# ## Topic Distribution

def get_topic_distribution(lda_model, vectorizer):

    # Get topic names
    topicnames = ["Topic" + str(i) for i in range(lda_model.n_components)]
    
    # Topic-Keyword Matrix
    df_topic_keywords = pd.DataFrame(lda_model.components_)

    # Assign Column and Index
    df_topic_keywords.columns = vectorizer.get_feature_names()
    df_topic_keywords.index = topicnames

    # View
    return df_topic_keywords


get_topic_distribution(lda_model_earthquake, earthquake_vectorizer)

get_topic_distribution(lda_model_fire, fire_vectorizer)

get_topic_distribution(lda_model_flood, flood_vectorizer)

get_topic_distribution(lda_model_hurricane, hurricane_vectorizer)


# Show top n keywords for each topic
def show_topics(vectorizer, lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    df_topic_keywords = pd.DataFrame(topic_keywords)
    df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
    df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
    return df_topic_keywords


show_topics(vectorizer=earthquake_vectorizer, lda_model=lda_model_earthquake, n_words=20)

show_topics(vectorizer=fire_vectorizer, lda_model=lda_model_fire, n_words=20)

show_topics(vectorizer=flood_vectorizer, lda_model=lda_model_flood, n_words=20)

show_topics(vectorizer=hurricane_vectorizer, lda_model=lda_model_hurricane, n_words=20)

#




