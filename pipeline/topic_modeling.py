import pandas as pd
import numpy as np
import re
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from gensim.parsing.preprocessing import preprocess_string, STOPWORDS 
from nltk.tokenize import word_tokenize
from gensim.models.phrases import Phrases, Phraser
from gensim.corpora import Dictionary
from gensim.models import LdaModel

random_seed = 10

#tags= ["parameters"]
upstream = []


# +
random_seed = 10

# ## Load the data

df_dev = pd.read_csv('../data/HumAID_data_v1.0/all_combined/all_dev.tsv', sep='\t')
df_train = pd.read_csv('../data/HumAID_data_v1.0/all_combined/all_train.tsv', sep='\t')
df_test = pd.read_csv('../data/HumAID_data_v1.0/all_combined/all_test.tsv', sep='\t')

df_dev.head()

df_all = pd.concat([df_dev, df_train, df_test])

df_all.isnull().sum()

df_all.dropna(inplace=True)
# -

df_all.head()

# +
#clean, tokenize text

# +
CUSTOM_STOP_WORDS = ['www','tinyurl','com', 'https', 'http', '&amp','amp', 'rt', 'bit', 'ly', 'bitly']

def clean_tokenize_text(tweet_df):
#     tweet_df['text'] = tweet_df['tweet_text'].astype(str)
    tweet_df['tokens'] = tweet_df['tweet_text'].apply(lambda x: preprocess_string(x))
    tweet_df['tokens'] = tweet_df['tokens'].apply(lambda x: [x[i] for i in range(len(x)) if x[i] not in CUSTOM_STOP_WORDS])
    return tweet_df

tweets_tokens = clean_tokenize_text(df_all)
tweets_tokens


# +
#create bigrams
def append_bigrams(tweet_df):
    phrases = Phraser(Phrases(tweet_df['tokens'],min_count=20,delimiter='_'))
    tweet_df['bigrams'] = tweet_df['tokens'].apply(lambda x: phrases[x])
    tweet_df['tokens'] = tweet_df['tokens']+tweet_df['bigrams']

    return tweet_df

tweets_bigrams = append_bigrams(tweets_tokens)
tweets_bigrams

# +
from gensim.corpora import Dictionary
from gensim.models import LdaModel

def find_topics(tokens, num_topics):
    
    dictionary = Dictionary(tokens)
    dictionary.filter_extremes(no_above=.4,keep_n=None)
     #words that represent more than 60% of the corpus
    # use the dictionary to create a bag of word representation of each document
    corpus = [dictionary.doc2bow(token) for token in tokens]
    # create gensim's LDA model 
    lda_model = LdaModel(corpus,
                         id2word=dictionary,
                         chunksize=2000,
                         passes=20,
                         iterations=400,
                         eval_every=None,
                         random_state=42,
                         alpha='auto',
                         eta='auto',
                         num_topics=num_topics)
    
    
    
    return lda_model.top_topics(corpus) 


find_topics(tweets_bigrams['tokens'], 10)


# +
def calculate_avg_coherence(topics):
    """
    Calculates the average coherence based on the top_topics returned by gensim's LDA model
    """
    x = 0
    for i, topic in enumerate(topics):
        x += topic[1]
    avg_topic_coherence = x/i
    
    return avg_topic_coherence


def plot_coherences_topics(tokens):
    """
    Creates a plot as shown above of coherence for the topic models created with num_topics varying from 2 to 10
    """
    # range of topics
    topics_range = range(2, 11, 1)
    model_results = {'Topics': [],'Coherence': []}
    for i in topics_range:
        model_topics = find_topics(tokens,i)
        model_results['Topics'].append(i)
        model_results['Coherence'].append(calculate_avg_coherence(model_topics))
    
    plt = pd.DataFrame(model_results).set_index('Topics').plot()

coherences_df = plot_coherences_topics(tweets_bigrams['tokens'])
# -










