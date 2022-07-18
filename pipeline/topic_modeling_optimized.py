import numpy as np
import pandas as pd
import re, nltk, spacy, gensim
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
nltk.download('punkt')
import tqdm

from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
# %matplotlib inline

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
# -

def text_pre_processing(x):
    # remove twitter handles (@user)
    text = re.sub(r'@[\S]*','',x)
    # remove web address
    text = re.sub(r'http[^\\S]*','',text)
    # remove special characters, numbers, punctuations
    text = text.replace(r'[^a-zA-Z#]', " ")
    # remove hashtag
    text = re.sub(r'#','',text)
    #split compound words
    text = re.sub( r"([A-Z])", r" \1", text)
    #remove short words
    text = ' '.join([w for w in text.split() if len(w)>3])
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
    tokenized.append(' '.join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc])) #if token.pos_ in allowed_postags]))
    return ''.join(tokenized)

# Do lemmatization keeping only Noun, Adj, Verb, Adverb
df_all['lemmatized'] = df_all['tweet_text_cleaned'].apply(lambda x: lemmatize(x, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']))

# -

tweet_corpus = df_all['lemmatized'].tolist()



# ## Count Vectorizer

# +
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words |= {"attach", "ahead", "rt",'www','tinyurl','com', 'https', 'http', '&amp','amp', 'rt', 'bit', 'ly', 'bitly'}

vectorizer = CountVectorizer(analyzer='word',       
                             min_df=10,                        # minimum reqd occurences of a word 
                             stop_words=stop_words,             # remove stop words
                             lowercase=False,                   # convert all words to lowercase
                             #token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                             # max_features=50000,             # max number of uniq words
                            )

tweets_vectorized = vectorizer.fit_transform(tweet_corpus)
# -



# ## LDA Model

# +
# Build LDA Model
lda_model = LatentDirichletAllocation(n_components=4,               # Number of topics
                                      max_iter=10,               # Max learning iterations
                                      learning_method='online',   
                                      random_state=random_seed,          # Random state
                                      batch_size=128,            # n docs in each learning iter
                                      evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                      n_jobs = -1,               # Use all available CPUs
                                     )
lda_output = lda_model.fit_transform(tweets_vectorized)

print(lda_model)  # Model attributes
# -

# ## Performance Stats

# +
# Log Likelyhood: Higher the better
print("Log Likelihood: ", lda_model.score(tweets_vectorized))

# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
print("Perplexity: ", lda_model.perplexity(tweets_vectorized))

# See model parameters
pprint(lda_model.get_params())
# -

# ## Grid Search Model Parameters

# +
# Define Search Param
search_params = {'n_components': [2, 4, 6, 8, 10], 'learning_decay': [.5, .7, .9]}

# Init the Model
lda = LatentDirichletAllocation()

# Init Grid Search Class
model = GridSearchCV(lda, param_grid=search_params)

# Do the Grid Search
model.fit(tweets_vectorized)
# -

# ## Best Model

# +
# Best Model
best_lda_model = model.best_estimator_

# Model Parameters
print("Best Model's Params: ", model.best_params_)

# Log Likelihood Score
print("Best Log Likelihood Score: ", model.best_score_)

# Perplexity
print("Model Perplexity: ", best_lda_model.perplexity(tweets_vectorized))
# -

# ## Grid Search Results

# +
# Get Log Likelyhoods from Grid Search Output
n_topics = [2, 4, 6, 8, 10]
log_likelyhoods_5 = [round(model.cv_results_['mean_test_score'][index]) for index, gscore in enumerate(model.cv_results_['params']) if gscore['learning_decay']==0.5]
log_likelyhoods_7 = [round(model.cv_results_['mean_test_score'][index]) for index, gscore in enumerate(model.cv_results_['params']) if gscore['learning_decay']==0.7]
log_likelyhoods_9 = [round(model.cv_results_['mean_test_score'][index]) for index, gscore in enumerate(model.cv_results_['params']) if gscore['learning_decay']==0.9]

# Show graph
plt.figure(figsize=(12, 8))
plt.plot(n_topics, log_likelyhoods_5, label='0.5')
plt.plot(n_topics, log_likelyhoods_7, label='0.7')
plt.plot(n_topics, log_likelyhoods_9, label='0.9')
plt.title("Choosing Optimal LDA Model")
plt.xlabel("Num Topics")
plt.ylabel("Log Likelyhood Scores")
plt.legend(title='Learning decay', loc='best')
plt.show()
# -

# ## Dominant Topics

# +
# Create Document - Topic Matrix
lda_output = best_lda_model.transform(tweets_vectorized)

# column names
topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)]

# index names
docnames = ["Doc" + str(i) for i in range(len(df_all))]

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
df_document_topics
# -



# ## Topic Distribution

# +
# Topic-Keyword Matrix
df_topic_keywords = pd.DataFrame(best_lda_model.components_)

# Assign Column and Index
df_topic_keywords.columns = vectorizer.get_feature_names()
df_topic_keywords.index = topicnames

# View
df_topic_keywords.head()


# +
# Show top n keywords for each topic
def show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords

topic_keywords = show_topics(vectorizer=vectorizer, lda_model=best_lda_model, n_words=15)        

# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
df_topic_keywords
# -

# ## Predict Topics from Text

# +
# Define function to predict topic for a given text document.
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def predict_topic(text, nlp=nlp):
    global sent_to_words #NEED TO DEFINE THIS
    global lemmatization #NEED TO DEFINE THIS

    # Step 1: Clean with simple_preprocess
    mytext_2 = list(sent_to_words(text))

    # Step 2: Lemmatize
    mytext_3 = lemmatization(mytext_2, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Step 3: Vectorize transform
    mytext_4 = vectorizer.transform(mytext_3)

    # Step 4: LDA Transform
    topic_probability_scores = best_lda_model.transform(mytext_4)
    topic = df_topic_keywords.iloc[np.argmax(topic_probability_scores), :].values.tolist()
    return topic, topic_probability_scores

# Predict the topic
mytext = ["Some text about christianity and bible"]
topic, prob_scores = predict_topic(text = mytext)
print(topic)
# -

# ## Get similar tweets

# +
from sklearn.metrics.pairwise import euclidean_distances

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def similar_documents(text, doc_topic_probs, documents = tweet_corpus, nlp=nlp, top_n=5, verbose=False):
    topic, x  = predict_topic(text)
    dists = euclidean_distances(x.reshape(1, -1), doc_topic_probs)[0]
    doc_ids = np.argsort(dists)[:top_n]
    if verbose:        
        print("Topic KeyWords: ", topic)
        print("Topic Prob Scores of text: ", np.round(x, 1))
        print("Most Similar Doc's Probs:  ", np.round(doc_topic_probs[doc_ids], 1))
    return doc_ids, np.take(documents, doc_ids)


# -

# Get similar documents
mytext = ["Some text about christianity and bible"]
doc_ids, docs = similar_documents(text=mytext, doc_topic_probs=lda_output, documents = tweet_corpus, top_n=1, verbose=True)
print('\n', docs[0][:500])




