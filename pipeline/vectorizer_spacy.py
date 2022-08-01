# # Building tweet vectorizer using spacy tokenizer

# +
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import spacy
import pickle

from nltk.stem.porter import PorterStemmer

import tools

# + tags=["parameters"]
upstream = []
random_seed = None
# -

# ## Read the train and test data

# +
df_train = pd.read_csv('../data/HumAID_data_v1.0/all_combined/all_train.tsv', sep='\t')
df_test = pd.read_csv('../data/HumAID_data_v1.0/all_combined/all_test.tsv', sep='\t')

df_train.dropna(inplace=True)
df_test.dropna(inplace=True)

df_train.sample(5)
# -
# ## Identify the most used locations in the training set and remove from the vocabulary - this may reduce the F1 score, but will also remove any data leakage

# #!python -m spacy download xx_ent_wiki_sm
# Fast and more accurate for LOC
nlp = spacy.load('xx_ent_wiki_sm')

# %%time
df_locs_train, locations_set_train = tools.get_locations(nlp, df_train)

# %%time
df_locs_test, locations_set_test = tools.get_locations(nlp, df_test)

# Visually inspect if the train and test locations are about the same - they look very close, remove from the vocabulary
print('Train: ', sorted(locations_set_train))
print()
print('Test: ', sorted(locations_set_test))

## Dump the stopwords
stopwords = locations_set_train | {"attach", "ahead", "rt"}
df_stopwords = pd.DataFrame(data=stopwords, columns=['stopword'])
df_stopwords.to_csv(product['stopwords'], index=False)
df_stopwords.sample(5)

# ## Build the tokenizer

# +
# Follow https://machinelearningknowledge.ai/complete-guide-to-spacy-tokenizer-with-examples/

# Run the first time
# #!python -m spacy download en_core_web_sm

nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()

# Add more stop words if needed - https://machinelearningknowledge.ai/tutorial-for-stopwords-in-spacy/
nlp.Defaults.stop_words |= set(df_stopwords['stopword'])

tokenizer = tools.Tokenizer(nlp, stopwords)

tokenizer.tokenize(
    'RT @HotshotWake: Good morning from Stewart Crossing Yukon up in Canada. Big fire day ahead. #canada #yukon #wildfire https://t.co/cSoymOMwJO')
# -

vectorizer = TfidfVectorizer(
    tokenizer=tokenizer.tokenize,
    strip_accents='unicode',
    min_df=10,
    ngram_range=(1, 2),
    max_features=10000,
    use_idf=True
)

# ## Test/validate with logistic regression

# +
# %%time
X_train = vectorizer.fit_transform(df_train['tweet_text'])
y_train = list(df_train['class_label'])

X_test = vectorizer.transform(df_test['tweet_text'])
y_test = list(df_test['class_label'])

X_train.shape
# -

# Prepate the logistic regression classifier
clf = LogisticRegression(solver='lbfgs', multi_class='auto', random_state=random_seed, max_iter=1000)
clf.fit(X_train, y_train)

# +
# %%time
# Predict on test, must be about the same
lr_test_preds = clf.predict(X_test)

# Score on the test data
lr_f1 = f1_score(y_test, lr_test_preds, average='macro')
print(lr_f1)
# -

# ## Dump the vectorizer and also the vocabulary

# +
with open(str(product['vectorizer']), 'wb') as f:
    pickle.dump(vectorizer, f)

with open(str(product['vocab']), 'wb') as f:
    pickle.dump(vectorizer.vocabulary_, f)
# -


