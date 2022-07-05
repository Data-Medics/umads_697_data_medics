# +
import pandas as pd
import numpy as np
import re
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.te
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split 
from tqdm import tqdm # Used to show a progress bar

#tags= ["parameters"]
upstream = []


# -

random_seed = 10

# ## Load the data

df_dev = pd.read_csv('../data/HumAID_data_v1.0/all_combined/all_dev.tsv', sep='\t')
df_train = pd.read_csv('../data/HumAID_data_v1.0/all_combined/all_train.tsv', sep='\t')
df_test = pd.read_csv('../data/HumAID_data_v1.0/all_combined/all_test.tsv', sep='\t')

df_dev.head()

df_all = pd.concat([df_dev, df_train, df_test])

df_all.isnull().sum()

df_all.dropna(inplace=True)

len(df_all)

#unique labels
df_all['class_label'].unique()

#class distributions
#will need to look at class weights
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

# +
# Most of the tokens are not alphanumeric ones, must consider some additional tokenization
print(len(ws_tokens))
print(len(alpha_ws_tokens))
print(len(alpha_re_tokens))

# Look at the most common, what do you see?
# Some stop words removal is needed
alpha_re_tokens.most_common(50)

# Not much difference for the simple tokens
ws_tokens.most_common(50)

# + endofcell="--"
# Plot the word distribution - it follows the power law
x = [*range(len(alpha_re_tokens))]
total = sum([b for _, b in alpha_re_tokens.most_common()])
y = [b / total for _, b in alpha_re_tokens.most_common()]

ax = plt.plot(x, y, '.')
plt.yscale('log')
plt.xscale('log')
# -
# --

# ## LDA 

# +
min_df = .005 # strong effect on determing which words get pruned from the vectorizer 
max_df = .999 # strong effect on determing which words get pruned from the vectorizer 
ngram_range = (1,1)

my_stop_words = text.ENGLISH_STOP_WORDS.union(['amp','rt','000'])
#word vectorizer, paramaters mostly set to defaults
vectorizer = CountVectorizer(strip_accents='ascii',
                lowercase=True,
                stop_words=my_stop_words,
                ngram_range = ngram_range,
                min_df = min_df,
                max_df = max_df,
                analyzer = 'word')

X_train = vectorizer.fit_transform(df_train['tweet_text'])
# -

tf_feature_names = vectorizer.get_feature_names_out()

vectorizer.get_feature_names_out()

X_train.shape

lda = LatentDirichletAllocation(n_components=20, 
                                doc_topic_prior=None, 
                                topic_word_prior=None, 
                                learning_method='batch', 
                                learning_decay=0.7, 
                                learning_offset=10.0, 
                                max_iter=10, 
                                batch_size=128, 
                                evaluate_every=- 1, 
                                total_samples=1000000.0, 
                                perp_tol=0.1, 
                                mean_change_tol=0.001, 
                                max_doc_update_iter=100, 
                                n_jobs=None, 
                                verbose=0, 
                                random_state=random_seed)
lda.fit(X_train)


def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(4, 5, figsize=(50, 40), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()


n_top_words = 20
plot_top_words(lda, tf_feature_names, n_top_words, "Topics in LDA model")
#definitely need to apply more rigourous text cleaning methods
#Look into genism and doc2bow


