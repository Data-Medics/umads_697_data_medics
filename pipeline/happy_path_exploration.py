# +
import pandas as pd
import numpy as np
import re

# NLP Imports
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split 
from tqdm import tqdm # Used to show a progress bar

# + tags=["parameters"]
upstream = []
lookback_days = None
random_seed = None
# -

# ## Read the train, dev and test data, do a very basic exploration

# +
df_dev = pd.read_csv('../data/HumAID_data_v1.0/all_combined/all_dev.tsv', sep='\t')
df_train = pd.read_csv('../data/HumAID_data_v1.0/all_combined/all_train.tsv', sep='\t')
df_test = pd.read_csv('../data/HumAID_data_v1.0/all_combined/all_test.tsv', sep='\t')

df_all = pd.concat([df_dev, df_train, df_test])

df_dev.dropna(inplace=True)
df_train.dropna(inplace=True)
df_test.dropna(inplace=True)

df_all.dropna(inplace=True)

df_dev.head()
# -
# Show the number of rows in the dev set
print("Dev rows:", df_dev.shape[0])
print("Train rows:", df_train.shape[0])
print("Test rows:", df_test.shape[0])
print("All rows:", df_all.shape[0])
# Show the number of labels/categories
len(set(df_all['class_label']))


# Show the unique labels/categories
list(df_all['class_label'].unique())

# Show the number of samples for each class - the data look like quite unbalanced,
#  validate in the train dataset, consider oversampling
# TODO: Consider classes with a minimum number of samples
Counter(df_all['class_label'])

# +
# Do some token exploration
ws_tokens = Counter()
alpha_ws_tokens = Counter()
alpha_re_tokens = Counter()

for tweet in tqdm(df_all['tweet_text']):
    for t in tweet.split(): ws_tokens.update([t])
    if re.fullmatch(r'\w+', t):
        alpha_ws_tokens.update([t])
        
    alpha_re_tokens.update(re.findall(r'\b\w+\b', tweet))
# -

# Most of the tokens are not alphanumeric ones, must consider some additional tokenization
print(len(ws_tokens))
print(len(alpha_ws_tokens))
print(len(alpha_re_tokens))

# Look at the most common, what do you see?
# Some stop words removal is needed
alpha_re_tokens.most_common(50)

# No much difference for the simple tokens
ws_tokens.most_common(50)

# +
# Plot the word distribution - it follows the power law
x = [*range(len(alpha_re_tokens))]
total = sum([b for _, b in alpha_re_tokens.most_common()])
y = [b / total for _, b in alpha_re_tokens.most_common()]

ax = plt.plot(x, y, '.')
plt.yscale('log')
plt.xscale('log')
# -

# ## Do a very basic vectorization and classification with logistic regression

# +
# %%time
# Next - add some classifiers
min_vocabulary = 200 # Changing this to higher value decreases the words in the vocabulary,
                     # consider finding the optimal value
                     # Also dropping this increases the F1 score

vectorizer = TfidfVectorizer(min_df=min_vocabulary, stop_words='english')
X_train = vectorizer.fit_transform(df_train['tweet_text'])
y_train = list(df_train['class_label'])
X_train.shape
# -

# Note: the default vocabularly is pretty small, try to enhance it
len(vectorizer.vocabulary_)

# Prepate the logistic regression classifier
clf = LogisticRegression(solver='lbfgs', multi_class='auto', random_state=random_seed, max_iter=1000)
clf.fit(X_train, y_train)

# +
# Do a dummy classifier 1 and 2
dummy_clf1 = DummyClassifier(strategy="uniform", random_state=random_seed)
dummy_clf1.fit(X_train, y_train)

dummy_clf2 = DummyClassifier(strategy="most_frequent", random_state=random_seed)
dummy_clf2.fit(X_train, y_train)

# +
# Predic on dev
X_dev = vectorizer.transform(df_dev['tweet_text'])
y_dev = list(df_dev['class_label'])

lr_tiny_dev_preds = clf.predict(X_dev)
rand_dev_preds = dummy_clf1.predict(X_dev)
mf_dev_preds = dummy_clf2.predict(X_dev)

# +
# Score on dev - not bad of out-of-the-box, must improve
lr_f1 = f1_score(y_dev, lr_tiny_dev_preds, average='macro')
rand_f1 = f1_score(y_dev, rand_dev_preds, average='macro')
mf_f1 = f1_score(y_dev, mf_dev_preds, average='macro')

print(lr_f1)
print(rand_f1)
print(mf_f1)

# +
# Predic on test, must be about the same
X_test = vectorizer.transform(df_test['tweet_text'])
y_test = list(df_test['class_label'])

lr_tiny_test_preds = clf.predict(X_test)
rand_test_preds = dummy_clf1.predict(X_test)
mf_test_preds = dummy_clf2.predict(X_test)

# +
# Score on test - about the same as dev
lr_f1 = f1_score(y_test, lr_tiny_test_preds, average='macro')
rand_f1 = f1_score(y_test, rand_test_preds, average='macro')
mf_f1 = f1_score(y_test, mf_test_preds, average='macro')

print(lr_f1)
print(rand_f1)
print(mf_f1)
# -

# ## Find out the f1 score for each class, find the best and worse

# +
df_try = df_test

X_try = vectorizer.transform(df_try['tweet_text'])
y_try = list(df_try['class_label'])

preds = clf.predict(X_try)

for label in set(df_try['class_label']):
    print(label, f1_score(y_try, preds, labels=[label], average='macro'))
# -

# ## Do bigrams see how this improves the F1 scores

bigram_vectorizer = TfidfVectorizer(stop_words='english', min_df=10, ngram_range=(1,2))
X_train = bigram_vectorizer.fit_transform(df_train['tweet_text'])
y_train = list(df_train['class_label'])
X_train.shape

# Prepate the logistic regression classifier
clf = LogisticRegression(solver='lbfgs', multi_class='auto', random_state=random_seed, max_iter=1000)
clf.fit(X_train, y_train)

# +
# Predic on test, must be about the same
X_test = bigram_vectorizer.transform(df_test['tweet_text'])
y_test = list(df_test['class_label'])

lr_tiny_test_preds = clf.predict(X_test)
# -

# Score on test - about the same as dev
lr_f1 = f1_score(y_test, lr_tiny_test_preds, average='macro')
print(lr_f1)

# +
# Find out the f1 score for each class, find the best and worse
df_try = df_test

X_try = bigram_vectorizer.transform(df_try['tweet_text'])
y_try = list(df_try['class_label'])

preds = clf.predict(X_try)

for label in set(df_try['class_label']):
    print(label, f1_score(y_try, preds, labels=[label], average='macro'))
# -

# ## Predict the disaster types as labels

# +
# Identify - what type of disasters and in what ratio
# Find the disaster locations - possibly some kind of topic modeling would be required. Find a location from text
#    https://medium.com/spatial-data-science/how-to-extract-locations-from-text-with-natural-language-processing-9b77035b3ea4

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

# Fire
df_fire_dev = pd.read_csv('../data/HumAID_data_v1.0/event_type/fire_dev.tsv', sep='\t')
df_fire_train = pd.read_csv('../data/HumAID_data_v1.0/event_type/fire_train.tsv', sep='\t')
df_fire_test = pd.read_csv('../data/HumAID_data_v1.0/event_type/fire_test.tsv', sep='\t')

df_fire_dev['disaster_type'] = 'fire'
df_fire_train['disaster_type'] = 'fire'
df_fire_test['disaster_type'] = 'fire'

# Flood
df_flood_dev = pd.read_csv('../data/HumAID_data_v1.0/event_type/flood_dev.tsv', sep='\t')
df_flood_train = pd.read_csv('../data/HumAID_data_v1.0/event_type/flood_train.tsv', sep='\t')
df_flood_test = pd.read_csv('../data/HumAID_data_v1.0/event_type/flood_test.tsv', sep='\t')

df_flood_dev['disaster_type'] = 'flood'
df_flood_train['disaster_type'] = 'flood'
df_flood_test['disaster_type'] = 'flood'

# Hurricane
df_hurricane_dev = pd.read_csv('../data/HumAID_data_v1.0/event_type/hurricane_dev.tsv', sep='\t')
df_hurricane_train = pd.read_csv('../data/HumAID_data_v1.0/event_type/hurricane_train.tsv', sep='\t')
df_hurricane_test = pd.read_csv('../data/HumAID_data_v1.0/event_type/hurricane_test.tsv', sep='\t')

df_hurricane_dev['disaster_type'] = 'hurricane'
df_hurricane_train['disaster_type'] = 'hurricane'
df_hurricane_test['disaster_type'] = 'hurricane'

df_disaster_dev = pd.concat([df_earthquake_dev, df_fire_dev, df_flood_dev, df_hurricane_dev])
df_disaster_train = pd.concat([df_earthquake_train, df_fire_train, df_flood_train, df_hurricane_train])
df_disaster_test = pd.concat([df_earthquake_test, df_fire_test, df_flood_test, df_hurricane_test])
# -

df_disaster_train.sample(5)

# Try predicting the disaster type of a corpus of tweets
bigram_vectorizer = TfidfVectorizer(stop_words='english', min_df=10, ngram_range=(1,2))
X_train = bigram_vectorizer.fit_transform(df_disaster_train['tweet_text'])
y_train = list(df_disaster_train['disaster_type'])
X_train.shape

clf = LogisticRegression(solver='lbfgs', multi_class='auto', random_state=random_seed, max_iter=1000)
clf.fit(X_train, y_train)

# +
X_test = bigram_vectorizer.transform(df_disaster_test['tweet_text'])
y_test = list(df_disaster_test['disaster_type'])

lr_tiny_test_preds = clf.predict(X_test)
# -

lr_f1 = f1_score(y_test, lr_tiny_test_preds, average='macro')
print(lr_f1)

# ## Predict each disaster type individually, use the micro-score in the F1 score

Counter(lr_tiny_test_preds)

# +
X_test = bigram_vectorizer.transform(df_earthquake_test['tweet_text'])
y_test = list(df_earthquake_test['disaster_type'])

lr_tiny_test_preds = clf.predict(X_test)
print(f1_score(y_test, lr_tiny_test_preds, average='micro'))
Counter(lr_tiny_test_preds)

# +
X_test = bigram_vectorizer.transform(df_hurricane_test['tweet_text'])
y_test = list(df_hurricane_test['disaster_type'])

lr_tiny_test_preds = clf.predict(X_test)
print(f1_score(y_test, lr_tiny_test_preds, average='micro'))
Counter(lr_tiny_test_preds)

# +
X_test = bigram_vectorizer.transform(df_flood_test['tweet_text'])
y_test = list(df_flood_test['disaster_type'])

lr_tiny_test_preds = clf.predict(X_test)
print(f1_score(y_test, lr_tiny_test_preds, average='micro'))
Counter(lr_tiny_test_preds)

# +
X_test = bigram_vectorizer.transform(df_fire_test['tweet_text'])
y_test = list(df_fire_test['disaster_type'])

lr_tiny_test_preds = clf.predict(X_test)
print(f1_score(y_test, lr_tiny_test_preds, average='micro'))
Counter(lr_tiny_test_preds)

# +
# TODO: Check if certain class of tweets is a better predictor of what kind of disaster is going on
class_labels = list(df_disaster_train['class_label'].unique())

for class_label in class_labels:
    df_tmp = df_disaster_train[df_disaster_train['class_label'] == class_label]

    bigram_vectorizer = TfidfVectorizer(stop_words='english', min_df=10, ngram_range=(1,2))
    X_train = bigram_vectorizer.fit_transform(df_tmp['tweet_text'])
    y_train = list(df_tmp['disaster_type'])
    #print(f'{class_label} train rows: ', X_train.shape[0])

    clf = LogisticRegression(solver='lbfgs', multi_class='auto', random_state=random_seed, max_iter=1000)
    clf.fit(X_train, y_train)

    X_test = bigram_vectorizer.transform(df_disaster_test['tweet_text'])
    y_test = list(df_disaster_test['disaster_type'])

    lr_tiny_test_preds = clf.predict(X_test)

    print(f'{class_label} F1 score: ', f1_score(y_test, lr_tiny_test_preds, average='macro'),
          ', train rows: ', X_train.shape[0])
# -
# ## Identify and plot the disaster locations from the tweets


# +
# Locations!!!
# -

import spacy
from spacy import displacy 

# +
# #!python -m spacy download en_core_web_sm
# #!python -m spacy download xx_ent_wiki_sm
# -

nlp = spacy.load('xx_ent_wiki_sm')

# +
# Go through the dev data and collect all the locations
locations = []

for _, row in tqdm(df_disaster_dev.iterrows()):
    doc = nlp(row['tweet_text'])    
    locations.extend([[row['tweet_id'], ent.text, ent.start, ent.end] for ent in doc.ents if ent.label_ in ['LOC']])
    
df_locs = pd.DataFrame(locations, columns=['TweetID', 'Location', 'start', 'end'])
# -

df_locs.sample(50)

# +
locations = Counter(df_locs['Location']).most_common(20)

df_locs_most_common = pd.DataFrame(locations, columns=['location', 'count'])

df_locs_most_common

# +
import geopandas as gpd 
import geopy

from geopy.extra.rate_limiter import RateLimiter

# +
locator = geopy.geocoders.Nominatim(user_agent='mygeocoder')
geocode = RateLimiter(locator.geocode, min_delay_seconds=1)

df_locs_most_common['address'] = df_locs_most_common['location'].apply(geocode)
# -

df_locs_most_common

df_locs_most_common['coordinates'] = df_locs_most_common['address'].apply(
    lambda loc: tuple(loc.point) if loc else None
)
df_locs_most_common[['latitude', 'longitude', 'altitude']] = pd.DataFrame(
    df_locs_most_common['coordinates'].tolist(), index=df_locs_most_common.index
)

df_locs_most_common

# +
import folium
from folium.plugins import FastMarkerCluster

folium_map = folium.Map(location=[59.338315,18.089960],
    zoom_start=2,
    tiles='CartoDB dark_matter')

FastMarkerCluster(data=list(zip(df_locs_most_common['latitude'].values,
                                df_locs_most_common['longitude'].values))).add_to(folium_map)
folium.LayerControl().add_to(folium_map)
folium_map
# -

# ## Use dense vectors and word embedding to test the predictions

### Word2vec exploration - check https://radimrehurek.com/gensim/models/word2vec.html
from nltk.corpus import stopwords


vectorizer = TfidfVectorizer(min_df=50, stop_words='english')
vectorizer.fit(df_disaster_train['tweet_text'])

analyzer = vectorizer.build_analyzer()

tokens = [analyzer(s) for s in list(df_disaster_train['tweet_text'])]
tokens[:2]

# +
# %%time
from gensim.models.word2vec import Word2Vec

quick_model = Word2Vec(sentences=tokens, vector_size=100, window=2,
                       min_count=10, workers=4, seed=random_seed)
# -
quick_model.wv.get_vector('quake')


quick_model.wv.similar_by_word("quake")[:10]


# +
# Build vectors for sentences, use for training a model
def generate_dense_features(tokenized_texts, model):
    result = []
    for tokens in tqdm(tokenized_texts):
        filtered_tokens = [t for t in tokens if t in model.wv.index_to_key] 
        if len(filtered_tokens) > 0:
            vec = np.mean(model.wv[filtered_tokens], axis=0) 
        else:
            vec = np.zeros(model.wv.vector_size)
        result.append(vec)
    return np.array(result)

X_train = generate_dense_features(tokens, quick_model)
# -

y_train = list(df_disaster_train['class_label'])

# Prepate the logistic regression classifier
clf = LogisticRegression(solver='lbfgs', multi_class='auto', random_state=random_seed, max_iter=1000)
clf.fit(X_train, y_train)

# Validate
tokens_test = [analyzer(s) for s in list(df_disaster_test['tweet_text'])]
tokens_test[:2]

# +
X_test = generate_dense_features(tokens_test, quick_model)
y_test = list(df_disaster_test['class_label'])

lr_tiny_test_preds = clf.predict(X_test)

f1_score(y_test, lr_tiny_test_preds, average='macro')

# +
## Idea - coref chain, timeline of events, extract actions done or advised by FEMA etc. Use NLP week 3 homework
# -

# ## Explain what words are the best predictors for certain prediction outcomes

## Explanation - use also the decision tree and information gain later on. Use NLP week 4 homework as an example
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline

# Train a model
# Do a bigrams and trigrams in the vocabulary
bigram_vectorizer = TfidfVectorizer(stop_words='english', min_df=10, ngram_range=(1,2))
X_train = bigram_vectorizer.fit_transform(df_train['tweet_text'])
y_train = list(df_train['class_label'])
X_train.shape

# Prepate the logistic regression classifier
clf = LogisticRegression(solver='lbfgs', multi_class='auto', random_state=random_seed, max_iter=1000)
clf.fit(X_train, y_train)

# +
# Predic on test, must be about the same
X_test = bigram_vectorizer.transform(df_test['tweet_text'])
y_test = list(df_test['class_label'])

lr_tiny_test_preds = clf.predict(X_test)

# Score on test
lr_f1 = f1_score(y_test, lr_tiny_test_preds, average='macro')
print(lr_f1)

# +
lr_pipe = make_pipeline(bigram_vectorizer, clf)

class_names=list(df_test['class_label'].unique())

explainer = LimeTextExplainer(class_names=class_names,
                              random_state=random_seed)

# +
test_row_df = df_test.sample(1, random_state=random_seed)
_, _, tweet_text, label = next(test_row_df.itertuples())

# tweet_text = "Non co-op with rescuers is no good. They are the best ones to decide if ppl have to be evacuated or given supplies. @vmoorthynow @aruna_sekhar @dhanyarajendran @InCrisisRelief @abhishekalex @MahalaxmiDas @RapidResponse #KeralaFloods"
# label = 'rescue_volunteering_or_donation_effort'

print(tweet_text)
print()
print('True label: ', label)
# -

print('Predicted label: ', lr_pipe.predict([tweet_text]))
print()
print('Class names: ', lr_pipe.classes_)
print()
print('Predicted confidence: ', lr_pipe.predict_proba([tweet_text])[0])

lr_explanation = explainer.explain_instance(tweet_text, lr_pipe.predict_proba, num_features=X_train.shape[1])

for feat, val in lr_explanation.as_list():
    print(f'Estimated weight of {val} on feature {feat}')

pd.DataFrame(data=lr_explanation.as_list(), columns=['feature', 'weight']).sort_values(by='weight', ascending=False)

# +
## TODO: Elaborate on the complete project path, add to the readme file, or the shared doc?
# -


