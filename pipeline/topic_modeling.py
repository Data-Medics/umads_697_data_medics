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
from gensim.models.ldamulticore import LdaMulticore
from tqdm import tqdm

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
# -

#



# +
def find_topics(tokens, num_topics):
    
    dictionary = Dictionary(tokens)
    dictionary.filter_extremes(no_above=.2,keep_n=None)
     #words that represent more than 80% of the corpus
    # use the dictionary to create a bag of word representation of each document
    corpus = [dictionary.doc2bow(token) for token in tokens]
    # create gensim's LDA model 
    lda_model = LdaModel(corpus,
                         id2word=dictionary,
                         chunksize=2000,
                         passes=20,
                         iterations=400,
                         eval_every=None,
                         random_state=random_seed,
                         alpha='auto',
                         eta='auto',
                         num_topics=num_topics)
    
    
    
    return lda_model.top_topics(corpus) 


find_topics(tweets_bigrams['tokens'], 10)


# +
##takes ~30-40 minutes to run
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
    for i in tqdm(topics_range):
        model_topics = find_topics(tokens,i)
        model_results['Topics'].append(i)
        model_results['Coherence'].append(calculate_avg_coherence(model_topics))
    
    plt = pd.DataFrame(model_results).set_index('Topics').plot()

coherences_df = plot_coherences_topics(tweets_bigrams['tokens'])
# -
# ## Set LDA_Model key parameters based on coherence analysis

dictionary = Dictionary(tweets_bigrams['tokens'])
dictionary.filter_extremes(no_above=.2,keep_n=None)
#words that represent more than 80% of the corpus
# use the dictionary to create a bag of word representation of each document
corpus = [dictionary.doc2bow(token) for token in tweets_bigrams['tokens']]
# create gensim's LDA model 
lda_model = LdaModel(corpus,
                    id2word=dictionary,
                    chunksize=2000,
                    passes=20,
                    iterations=400,
                    eval_every=None,
                    random_state=random_seed,
                    alpha='auto',
                    eta='auto',
                    num_topics=4)


# ## Get dominant topics, percent contribution and keywords

# +
tweets=tweets_bigrams['tweet_text'].tolist()


def format_topics_sentences(ldamodel=None, corpus=corpus, texts=tweets):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in tqdm(enumerate(ldamodel[corpus])):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=tweets)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic.head(20)
# -

# ## Get the most representative sentence for each topic

# +
# Display setting to show more characters in column
pd.options.display.max_colwidth = 100

sent_topics_sorteddf_mallet = pd.DataFrame()
sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in tqdm(sent_topics_outdf_grpd):
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=False).head(1)], 
                                            axis=0)

# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]

# Show
sent_topics_sorteddf_mallet.head(10)
# -

# ## Frequency distribution of topics

# +
doc_lens = [len(d) for d in df_dominant_topic.Text]

# Plot
plt.figure(figsize=(16,7), dpi=160)
plt.hist(doc_lens, bins = 1000, color='navy')
plt.text(750, 100, "Mean   : " + str(round(np.mean(doc_lens))))
plt.text(750,  90, "Median : " + str(round(np.median(doc_lens))))
plt.text(750,  80, "Stdev   : " + str(round(np.std(doc_lens))))
plt.text(750,  70, "1%ile    : " + str(round(np.quantile(doc_lens, q=0.01))))
plt.text(750,  60, "99%ile  : " + str(round(np.quantile(doc_lens, q=0.99))))

plt.gca().set(xlim=(0, 1000), ylabel='Number of Documents', xlabel='Document Word Count')
plt.tick_params(size=16)
plt.xticks(np.linspace(0,1000,9))
plt.title('Distribution of Document Word Counts', fontdict=dict(size=22))
plt.show()
# -


# ## Distribution of Document Word Counts by Dominant Topic

# +
import seaborn as sns
import matplotlib.colors as mcolors
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

fig, axes = plt.subplots(2,2,figsize=(16,14), dpi=160, sharex=True, sharey=True)

for i, ax in tqdm(enumerate(axes.flatten())):    
    df_dominant_topic_sub = df_dominant_topic.loc[df_dominant_topic.Dominant_Topic == i, :]
    doc_lens = [len(d) for d in df_dominant_topic_sub.Text]
    ax.hist(doc_lens, bins = 1000, color=cols[i])
    ax.tick_params(axis='y', labelcolor=cols[i], color=cols[i])
    sns.kdeplot(doc_lens, color="black", shade=False, ax=ax.twinx())
    ax.set(xlim=(0, 1000), xlabel='Document Word Count')
    ax.set_ylabel('Number of Documents', color=cols[i])
    ax.set_title('Topic: '+str(i), fontdict=dict(size=16, color=cols[i]))

fig.tight_layout()
fig.subplots_adjust(top=0.90)
plt.xticks(np.linspace(0,1000,9))
fig.suptitle('Distribution of Document Word Counts by Dominant Topic', fontsize=22)
plt.show()
# -

# ## Word Clouds of Top N Keywords in Each Topic

# +
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = lda_model.show_topics(formatted=False)

fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

for i, ax in tqdm(enumerate(axes.flatten())):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()
# -

# ## Word count and importance of topic keywords

# +
bigrams=tweets_bigrams['bigrams'].tolist()

from collections import Counter
topics = lda_model.show_topics(formatted=False)
data_flat = [w for w_list in bigrams for w in w_list]
counter = Counter(data_flat)

out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i , weight, counter[word]])

df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        

# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(2, 2, figsize=(16,10), sharey=True, dpi=160)
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in tqdm(enumerate(axes.flatten())):
    ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
    ax.set_ylabel('Word Count', color=cols[i])
    ax_twin.set_ylim(0, 0.07); ax.set_ylim(0, 15000)
    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)    
fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    
plt.show()
# -

# ## t-sne clustering

# +
# Get topic weights and dominant topics ------------
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook

# Get topic weights
topic_weights = []
for i, row_list in enumerate(lda_model[corpus]):
    topic_weights.append([w for i,w in row_list])

# Array of topic weights    
arr = pd.DataFrame(topic_weights).fillna(0).values

# Keep the well separated points (optional)
arr = arr[np.amax(arr, axis=1) > 0.35]

# Dominant topic number in each doc
topic_num = np.argmax(arr, axis=1)

# tSNE Dimension Reduction
tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
tsne_lda = tsne_model.fit_transform(arr)

# Plot the Topic Clusters using Bokeh
output_notebook()
n_topics = 4
mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), 
              plot_width=900, plot_height=700)
plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])
show(plot)
# -

# ## pyLDAVis

import pyLDAvis.gensim
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
vis




