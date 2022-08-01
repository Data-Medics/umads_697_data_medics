# # Building tweet vectorizer using a standard TweetTokenizer

# +
import pandas as pd
from sklearn.metrics import f1_score
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer
from sklearn.linear_model import LogisticRegression

# + tags=["parameters"]
upstream = ['vectorizer']
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
# ## Build and train the vectorizer

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

# +
# from sklearn.decomposition import TruncatedSVD

# svd = TruncatedSVD(n_components=500, n_iter=30, random_state=42)

# +
# %%time
X_train = vectorizer.fit_transform(df_train['tweet_text'])
#X_train_reduced = svd.fit_transform(X_train)
y_train = list(df_train['class_label'])

X_test = vectorizer.transform(df_test['tweet_text'])
#X_test_reduced = svd.transform(X_test)
y_test = list(df_test['class_label'])

print('Categories: ', list(set(y_train)))
print('Vectorizer rows and columns: ', X_train.shape)
print()
# -

# ## Test/validate with logistic regression

# %%time
# Prepate the logistic regression classifier
clf = LogisticRegression(solver='lbfgs', multi_class='auto', random_state=random_seed, max_iter=1000)
clf.fit(X_train, y_train)

# %%time
# Predict on test
lr_test_preds = clf.predict(X_test)
# Score on the test data
lr_f1 = f1_score(y_test, lr_test_preds, average='macro')
print('F1 score on the test data: ', lr_f1)

# ## Persist the vectorizer to be used downstream

# +
with open(str(product['vectorizer']), 'wb') as f:
    pickle.dump(vectorizer, f)

with open(str(product['vocab']), 'wb') as f:
    pickle.dump(vectorizer.vocabulary_, f)
