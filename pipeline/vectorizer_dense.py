# # Building tweet vectorizer using dense vectors (Word2Vec)

# +
import pandas as pd
from sklearn.metrics import f1_score
import pickle

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer
from gensim.models.word2vec import Word2Vec
from sklearn.linear_model import LogisticRegression

from tools import generate_dense_features

# + tags=["parameters"]
upstream = []
random_seed = 42
# -

# ## Read the train and test data

# +
df_train = pd.read_csv('../data/HumAID_data_v1.0/all_combined/all_train.tsv', sep='\t')
df_test = pd.read_csv('../data/HumAID_data_v1.0/all_combined/all_test.tsv', sep='\t')

df_train.dropna(inplace=True)
df_test.dropna(inplace=True)

df_train.sample(5)
# -
# ## Build and train the vectorizer and obtain the respective analyzer

# +
tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)

vectorizer = TfidfVectorizer(
        tokenizer=tokenizer.tokenize, #tokenize_text,
        strip_accents='unicode',
        ngram_range=(1, 2),
        max_df=0.90,
        min_df=1,
        max_features=10000,
        use_idf=True
    )
# -

# %%time
vectorizer.fit(df_train['tweet_text'])

analyzer = vectorizer.build_analyzer()

tokens = [analyzer(s) for s in list(df_train['tweet_text'])]
tokens[100]

# ## Init the dense vectorizer

quick_model = Word2Vec(sentences=tokens, vector_size=100, window=2,
                       min_count=10, workers=4, seed=random_seed)

quick_model.wv.get_vector('quake')

# ## Build vectors for the sentences, use for training a model

# Train
X_train = generate_dense_features(tokens, quick_model)
y_train = list(df_train['class_label'])

# Test
tokens_test = [analyzer(s) for s in list(df_test['tweet_text'])]
tokens_test[100]

X_test = generate_dense_features(tokens_test, quick_model)
y_test = list(df_test['class_label'])

# ## Train the LR model, score using the test

clf = LogisticRegression(solver='lbfgs', multi_class='auto', random_state=random_seed, max_iter=1000)
clf.fit(X_train, y_train)

lr_test_preds = clf.predict(X_test)
f1_lr = f1_score(y_test, lr_test_preds, average='macro')
print(f1_lr)


