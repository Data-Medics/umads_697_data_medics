# + tags=[]



# + tags=["parameters"]
# # + tags=["parameters"]
# declare a list tasks whose products you want to use as inputs
upstream = ['vectorizer_countVec','topic_modeling_disaster_type_final']
random_seed = 42
from tools_rjg import *
import pandas as pd
import re, nltk, spacy, gensim
nltk.download('punkt')
import pyLDAvis.sklearn
import pickle
import locations as loc
import os
# + tags=["injected-parameters"]
# This cell was injected automatically based on your stated upstream dependencies (cell above) and pipeline.yaml preferences. It is temporary and will be removed when you save this notebook
upstream = {
    "topic_modeling_disaster_type_final": {
        "nb": "C:\\Users\\gillrobe\\DataScience\\umads_697_data_medics\\pipeline\\output\\topic_modeling_disaster_type_final.ipynb",
        "lda_model_earthquake": "C:\\Users\\gillrobe\\DataScience\\umads_697_data_medics\\pipeline\\output\\lda_model_earthquake.pkl",
        "lda_model_fire": "C:\\Users\\gillrobe\\DataScience\\umads_697_data_medics\\pipeline\\output\\lda_model_fire.pkl",
        "lda_model_flood": "C:\\Users\\gillrobe\\DataScience\\umads_697_data_medics\\pipeline\\output\\lda_model_flood.pkl",
        "lda_model_hurricane": "C:\\Users\\gillrobe\\DataScience\\umads_697_data_medics\\pipeline\\output\\lda_model_hurricane.pkl",
        "lda_topics_disaster_type": "C:\\Users\\gillrobe\\DataScience\\umads_697_data_medics\\pipeline\\output\\lda_topics_disaster_type.csv",
    },
    "vectorizer_countVec": {
        "nb": "C:\\Users\\gillrobe\\DataScience\\umads_697_data_medics\\pipeline\\output\\vectorizer_countVec.ipynb",
        "vectorizer": "C:\\Users\\gillrobe\\DataScience\\umads_697_data_medics\\pipeline\\output\\vectorizer_countVec.pkl",
    },
}
product = {
    "nb": "C:\\Users\\gillrobe\\DataScience\\umads_697_data_medics\\pipeline\\output\\topic_modeling_disaster_type_results.ipynb"
}


# + tags=[]
df_all = generate_disaster_type_dataframe(disaster_types = ('earthquake', 'fire', 'flood', 'hurricane')
                                          , dev_train_test= ('dev', 'train', 'test'))

# + tags=[]
with open("output/vectorizer_countVec.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# + tags=[]
with open("output/lda_model_earthquake.pkl", "rb") as f:
    lda_model_earthquake = pickle.load(f)
with open("output/lda_model_fire.pkl", "rb") as f:
    lda_model_fire = pickle.load(f)
with open("output/lda_model_flood.pkl", "rb") as f:
    lda_model_flood = pickle.load(f)
with open("output/lda_model_hurricane.pkl", "rb") as f:
    lda_model_hurricane = pickle.load(f)


# + tags=[]
df_all['tweet_text_cleaned'] = df_all['tweet_text'].apply(lambda x: tweet_preprocessing(x))
df_all['lemmatized'] = df_all['tweet_text_cleaned'].apply(lambda x: lemmatize_tweet_text(x, allowed_postags=('NOUN', 'ADJ', 'VERB', 'ADV')))

# + tags=[]
df_all.sample(100)

# + tags=[]
earthquake_text_vectorized = vectorizer.transform(list(df_all[df_all['disaster_type']=='earthquake']['lemmatized']))
fire_text_vectorized = vectorizer.transform(list(df_all[df_all['disaster_type']=='fire']['lemmatized']))
flood_text_vectorized = vectorizer.transform(list(df_all[df_all['disaster_type']=='flood']['lemmatized']))
hurricane_text_vectorized = vectorizer.transform(list(df_all[df_all['disaster_type']=='hurricane']['lemmatized']))

# + [markdown] tags=[]
# ## Performance Stats

# + tags=[]
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

# + tags=[]
get_dominant_topics(lda_model_earthquake,earthquake_text_vectorized)

# + tags=[]
get_dominant_topics(lda_model_fire,fire_text_vectorized)

# + tags=[]
get_dominant_topics(lda_model_flood,flood_text_vectorized)

# + tags=[]
get_dominant_topics(lda_model_hurricane, hurricane_text_vectorized)

# + [markdown] tags=[]
# ## Get Topic Distribution

# + tags=[]
get_topic_distribution(lda_model_earthquake, vectorizer)


# + tags=[]
get_topic_distribution(lda_model_fire, vectorizer)
# + tags=[]
get_topic_distribution(lda_model_flood, vectorizer)


# + tags=[]
get_topic_distribution(lda_model_hurricane, vectorizer)

# + [markdown] tags=[]
# ## Show Top N Topics

# + tags=[]
show_topics(vectorizer, fitted_lda_model=lda_model_earthquake, n_words=20)

# + tags=[]
show_topics(vectorizer, fitted_lda_model=lda_model_fire, n_words=20)

# + tags=[]
show_topics(vectorizer, fitted_lda_model=lda_model_flood, n_words=20)

# + tags=[]
show_topics(vectorizer, fitted_lda_model=lda_model_hurricane, n_words=20)

# + [markdown] tags=[]
# ## Extract most representative sentence for each topic

# + tags=[]
pd.options.display.max_colwidth = 1000

# + tags=[]
earthquake_top_sentence = extract_most_representative_sentence(lda_model_earthquake, earthquake_text_vectorized, df_all[df_all['disaster_type']=='earthquake'])
for topic in range(lda_model_earthquake.n_components):
    print('----------------------')
    print(f'Topic_{topic}', earthquake_top_sentence[f'Topic_{topic}'][1]['tweet_text'])

# + tags=[]
fire_top_sentence = extract_most_representative_sentence(lda_model_fire, fire_text_vectorized, df_all[df_all['disaster_type']=='fire'])
for topic in range(lda_model_fire.n_components):
    print('----------------------')
    print(f'Topic_{topic}', fire_top_sentence[f'Topic_{topic}'][1]['tweet_text'])

# + tags=[]



# + tags=[]
flood_top_sentence = extract_most_representative_sentence(lda_model_flood, flood_text_vectorized, df_all[df_all['disaster_type']=='flood'])
for topic in range(lda_model_flood.n_components):
    print('----------------------')
    print(f'Topic_{topic}', flood_top_sentence[f'Topic_{topic}'][1]['tweet_text'])

# + tags=[]



# + tags=[]
hurricane_top_sentence = extract_most_representative_sentence(lda_model_hurricane, hurricane_text_vectorized, df_all[df_all['disaster_type']=='hurricane'])
for topic in range(lda_model_hurricane.n_components):
    print('----------------------')
    print(f'Topic_{topic}', hurricane_top_sentence[f'Topic_{topic}'][1]['tweet_text'])

# +
disaster_types = ['earthquake', 'fire', 'flood', 'hurricane']
def combine_top_sentences():
    df_combined_top_sent = pd.DataFrame()
    
    for disaster in disaster_types:
        
        if disaster == 'earthquake':
            n_components = lda_model_earthquake.n_components
            for i in range(n_components):
                topic_string = str(f'Topic_{i}')
                df = earthquake_top_sentence[topic_string][1][['tweet_text','disaster_type']]
                df['topic'] = f'Topic {i+1}'
                df_combined_top_sent = pd.concat([df_combined_top_sent,df])
        elif disaster =='fire':
            n_components = lda_model_fire.n_components
            for i in range(n_components):
                topic_string = str(f'Topic_{i}')
                df = fire_top_sentence[topic_string][1][['tweet_text','disaster_type']]
                df['topic'] = f'Topic {i+1}'
                df_combined_top_sent = pd.concat([df_combined_top_sent,df])

        elif disaster =='flood':
            n_components = lda_model_flood.n_components
            for i in range(n_components):
                topic_string = str(f'Topic_{i}')
                df = flood_top_sentence[topic_string][1][['tweet_text','disaster_type']]
                df['topic'] = f'Topic {i+1}'
                df_combined_top_sent = pd.concat([df_combined_top_sent,df])

        elif disaster =='hurricane':
            n_components = lda_model_hurricane.n_components
            for i in range(n_components):
                topic_string = str(f'Topic_{i}')
                df = hurricane_top_sentence[topic_string][1][['tweet_text','disaster_type']]
                df['topic'] = f'Topic {i+1}'
                df_combined_top_sent = pd.concat([df_combined_top_sent,df])

    return df_combined_top_sent


# -

combined_top_sentences = combine_top_sentences()
combined_top_sentences.to_csv('..\\blog_post\\blog_data\\lda_top_sentence.csv',header=True)

# + tags=[]



# + tags=[]
pyLDAvis.enable_notebook()
panel_earthquake = pyLDAvis.sklearn.prepare(lda_model_earthquake, earthquake_text_vectorized, vectorizer, mds='tsne',sort_topics=False)
panel_fire = pyLDAvis.sklearn.prepare(lda_model_fire, fire_text_vectorized, vectorizer, mds='tsne',sort_topics=False)
panel_flood = pyLDAvis.sklearn.prepare(lda_model_flood, flood_text_vectorized, vectorizer, mds='tsne',sort_topics=False)
panel_hurricane = pyLDAvis.sklearn.prepare(lda_model_hurricane, hurricane_text_vectorized, vectorizer, mds='tsne',sort_topics=False)
panel

# + tags=[]
pyLDAvis.save_html(panel_earthquake, '..\\blog_post\\blog_data\\lda_earthquake_pyldavis.html')
pyLDAvis.save_html(panel_fire, '..\\blog_post\\blog_data\\lda_fire_pyldavis.html')
pyLDAvis.save_html(panel_flood, '..\\blog_post\\blog_data\\lda_flood_pyldavis.html')
pyLDAvis.save_html(panel_hurricane, '..\\blog_post\\blog_data\\lda_hurricane_pyldavis.html')
# -


