import os
import pandas as pd
import pickle
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

# + tags=["parameters"]
upstream = []
random_seed = 42

train_data_path = "../data/HumAID_data_v1.0/all_combined/all_train.tsv"
dev_data_path = "../data/HumAID_data_v1.0/all_combined/all_dev.tsv"
test_data_path = "../data/HumAID_data_v1.0/all_combined/all_test.tsv"
# -

if not os.path.exists(os.path.join(os.getcwd(), "output", "fitted_models")):
    os.makedirs(os.path.join(".", "output", "fitted_models"))

# +
# load data
df_dev = pd.read_csv(dev_data_path, sep='\t')
df_train = pd.read_csv(train_data_path, sep='\t')
df_test = pd.read_csv(test_data_path, sep='\t')

df_dev.dropna(inplace=True)
df_train.dropna(inplace=True)
df_test.dropna(inplace=True)
# -

# instantiate the tokenizer
# TODO: import from tools.py once we agree upon a final tokenizer
tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)


# +
# build the custom tokenizer function

def get_vectorizer_custom():
    vectorizer = TfidfVectorizer(
        tokenizer=tokenizer.tokenize, #tokenize_text,
        strip_accents='unicode',
        ngram_range=(1, 2),
        max_df=0.90,
        min_df=1,
        max_features=10000,
        use_idf=True
    )
    return vectorizer


vectorizer = get_vectorizer_custom()

# +
X_train = vectorizer.fit_transform(df_train['tweet_text'])
y_train = list(df_train['class_label'])

X_test = vectorizer.transform(df_test['tweet_text'])
y_test = list(df_test['class_label'])
# -

clf = LogisticRegression(solver='lbfgs', multi_class='auto', random_state=random_seed, max_iter=1000)
clf.fit(X_train, y_train)

# +
# Predict on test
lr_test_preds = clf.predict(X_test)

# Score on the test data
lr_f1 = f1_score(y_test, lr_test_preds, average='macro')

print(lr_f1)


# -
filename = "./output/fitted_models/lr_model.pkl"
pickle.dump(clf, open(filename, 'wb'))


filename = "./output/fitted_models/vectorizer.pkl"
pickle.dump(vectorizer, open(filename, 'wb'))
