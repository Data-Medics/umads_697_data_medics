import itertools
import streamlit as st
import os
import pickle
import pandas as pd
import sys
import re
import datetime
import spacy
from streamlit import components
import altair as alt

sys.path.insert(0, "./pipeline")
import locations as loc

# CSS to inject contained in a string
# https://docs.streamlit.io/knowledge-base/using-streamlit/hide-row-indices-displaying-dataframe
hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

def check_locs(locs, search_locs):
    return any([s in locs for s in search_locs])

def make_str(list_of_verbs):
    list_of_verbs = [a.lower() for a in list_of_verbs if a != "all"]
    if len(list_of_verbs) == 1:
        return list_of_verbs[0]
    else:
        return ' or '.join(set(list_of_verbs))
    
    
# from https://discuss.streamlit.io/t/how-to-add-extra-lines-space/2220/7
def v_spacer(height, sb=False) -> None:
    for _ in range(height):
        if sb:
            st.sidebar.write('\n')
        else:
            st.write('\n')


st.set_page_config(layout="wide")
st.title('DATA MEDICS')
st.header('Disaster Tweet Pipeline')
st.markdown('Created By:')
st.markdown('**Tyler Zupan**'')
st.markdown('**Robert Gillin**'')
st.markdown('**Miroslav Boussarov**'')


tabs_list = ["Project Summary", "Topic Modeling", "Tweet Classification", "Recent Tweets - Comparative Analysis", 
"Recent Tweets - Individual Analysis", "Real Time Tweet Analysis", "Final Summary", "Additional Information"]
tab_1, tab_2, tab_3, tab_4, tab_5, tab_6, tab_7, tab_8 = st.tabs(tabs_list)
with tab_1:
    st.header(tabs_list[0])
    st.subheader("Overview")
    """Quality information is incredibly difficult to obtain following a major natural disaster and finding specific, actionable information can make a huge difference to those affected.  Lots of pertinent information is posted on Twitter following natural disasters but it can be extremely challenging to sift through hundreds of thousands of tweets for the desired information.  To help solve this problem we built a one-stop-natural-disaster-information-shop.  Powered by our machine learning models, we make it easy to filter tweets into specific categories based on the information the user is trying to find.  We use natural language processing and topic models to search Twitter for tweets specific to different types of natural disasters.  We provide a wide variety of carefully curated, useful information ranging from geo-spatial analysis to understand the exact locations and severity of the natural disaster to parsing the Twitterverse for actionable behaviors.  The period following a natural disaster is incredibly difficult and dangerous for all those involved - our tool aims to make the recovery process easier and more efficient with better outcomes for those affected."""
    
    st.subheader("Historical Data")
    """All models in this project were trained on a data set collected and curated by HumAid.  The data consists of ~70,000 tweets across 20 natural disasters spanning from 2016 to 2018.  The tweets were labeled by human assistants and are classified as belonging to one of the following 10 categories:
    
    Caution and advice
    Displaced people and evacuations
    Dont know cant judge
    Infrastructure and utility damage
    Injured or dead people
    Missing or found people
    Not humanitarian
    Other relevant information
    Requests or urgent needs
    Rescue volunteering or donation effort
    Sympathy and support
    
A sample of the data can be seen here:
"""
    sampled_data = pd.read_csv(os.path.join(loc.blog_data, "train_data_sample.csv"))
    st.write(sampled_data.sample(5))

    "More information about the data set can be found [here](https://crisisnlp.qcri.org/humaid_dataset.html#)."
    
    st.subheader("Real Time Data")
    """
    However, giving people critical information about a disaster that happened several years ago is of limited value.  For our project to truly help, we needed to apply our models and analysis to live Twitter data around current natural disasters.  To do this we utilized [Tweepy](https://www.tweepy.org/) and built an app that pulls real time tweats based on our disaster specific search terms which we then use as inputs into our models.
    """
    st.subheader("Project Plan")
    """
    In the beginning, we scoped our project as a fairly simple pipeline: we would train several models from labeled disaster tweet data, then use these trained models to produce 
    actionable insights and information for those involved in current natural disasters. Along the way we used NLP-centric unsupervised learning techniques to buttress our work - for example, 
    we used the most relevant disaster related keywords from our topic models as query search terms to gather real time Twitter data for current disasters.  Our original project plan layout looked like
    the following:
    """
    st.image(os.path.join(loc.blog_data, "model_pipeline.png"), caption=None)


    """
    The information derived from these topic models, as well as the additional models trained on the labeled data is then applied to real-time Tweets to generate useful information.  The end product, an app built in Streamlit was planned to do the following:
    """
    st.image(os.path.join(loc.blog_data, "streamlit_app.png"), caption=None)

    st.subheader("Blog Layout")
    """
    The diagrams above describe our initial plan - what we hoped to accomplish and why we felt the output would be useful 
    to people in crises, in addition to being an interesting data science problem to solve. The rest of the blog, as laid 
    out on the following tabs, will walk you through our work - starting with raw tweet data all the way to producing actionable 
    recommendations for current natural disasters and next steps.
    * Topic Modeling
    * Tweet Classification
    * Recent Tweets - Comparative Analysis
    * Recent Tweets - Individual Analysis
    * Real Time Tweet Analysis
    * Final Summary
    * Additional Information
    """
    
    
with tab_2:
    st.header(tabs_list[1])
    st.subheader("Overview")
    """
        In order to provide actionable recommendations during times of disaster, we needed a way to efficiently query 
        recent and relevant disaster tweets from twitter. To do so, we developed a function to extract tweets from the 
        Twitter API based on a query of keywords. The keywords we used to build this query were a direct product of the 
        Latent Dirichlet Allocation (LDA) topic modeling analysis we conducted on the HumAid disaster tweets dataset.
    """
    st.subheader("Topic Modeling Exploratory Analysis")
    st.markdown("**Token Exploration**")
    """
        To gain some initial familiarity with the disaster tweet text, we performed some simple token exploration.
    More specifically we looked at how many tokens were composed of word characters versus tokens comprised of non-word 
    characters. Tweet text can be very messy, and we would eventually need to decide how to handle hashtags, symbols, 
    retweets and mentions.
    """
    st.subheader("Model Design")
    st.markdown("**Text Pre-Processing**")
    """
        After a few initial iterations, it became clear that the success and performance of our topic model
    would largely depend on how well we were able to pre-process the disaster tweet text. This naturally became an iterative process
    as we adjusted the following parameters and inputs, measuring topic quality at each iteration:
    1. Stopword removal
    2. Tokenization technique (single tokens, bigrams, trigams)
    3. Splitting of compound words
    4. Inclusion and exclusion of hashtags, twitter handles (@user), web addresses, non-ascii characters
    5. Word length
    """
    st.markdown("**Lemmatization & Vectorization**")
    """
        Lemmatization is the process of reducing various inflectional forms of a word to it's base form. Stemming is a 
        more crude approach in which the end or the beginning of a word is cut off to reduce the word to it's root. 
        We chose lemmatization over stemming because we wanted to retain the morphological structure of words, knowing 
        this would be a more robust way to capture topics effectively. 
    
    We used the spaCy [Lemmatizer](https://spacy.io/api/lemmatizer) pipeline component to transform our pre-processed tweet text into it's lemmatized form.
    One challenge we encountered when initially building out our topic model, was that our topics were being focused around
    specific events and locations. This was undesirable as we wanted our topics to be generalizable to future disasters. spaCy's 
    lemmatzier gave us the flexibility to filter out unwanted part-of-speech tags. We chose to only include
    nouns, adjectives, verbs and adverbs in our final lemmatized text. In some cases there were still location names that slipped
    through our pre-processing steps. For these unique cases, we added these words to our list of stopwords.
    
    
    For our text vectorization, we needed to convert our lemmatized text into a bag-of-words representation which the LDA model
    could use. We used the sklearn [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) 
    class to produce a sparse representation of token counts. The CountVectorizer class conveniently let us set our ngram_range to (1,2) along 
    with filtering out our pre-defined stopwords.
    """
    st.markdown("**Model Selection and Training**")
    """
        We used the sklearn [LatentDirichletAllocation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)
    class for our topic model. Initially we had built the topic model on the entire disaster tweet dataset. However, it became clear that
    the topics we were generating were suboptimal from a qualitative review perspective. Additionally, the topics were 
    being blended across the various disaster types leading to poor interpretability. To address this, we decided to train a 
    separate topic model for each disaster type. While this introduced additional algorithmic complexity, it turned out 
    to be the right approach, as our topics became much more concise and easily interpretable. 
    
    The single key parameter for the LDA model is the "n_components" parameter. After some initial experimentation,
    it seemed that an n_components value between 2-4 was returning relatively concise topics. Eventually we would perform a more
    robust hyperparameter tuning exercise to systematically estimate the optimal number of topics, but initially we set n_components = 3,
    so we could continue with our development work. From here, we fine-tuned various aspects of our topic modeling pipeline using log-liklihood
    scores, perplexity scores and general topic interpretability to guide our refinements to the model. 
    """
    st.markdown("**Hyperparameter Tuning**")
    """  
        The final step before deploying the LDA model was to estimate the optimal number of topics for each disaster type LDA model. We developed a script
    that performs a grid search exersice, varying the number of topics from 2-50 across each disaster type LDA model. We then computed the
    log-liklihood and coherence scores for each configuration. When calculating coherence, we used the 'u_mass' measurement as opposed to
    the 'c_v' measurement. The 'c_v' measurement is known to produce the most reliable results, but opted to use the 'u_mass' measurement
    to speed up the computation. In the future, if we have more computing resources available at our disposal, we would choose to run the 'c_v' coherence measurement for better accuracy.
     The full grid search took approximately 4 hrs to run. See the figure below for the results.
    """
    coherence_df = pd.read_csv(os.path.join(loc.blog_data, 'lda_coherence_results.csv'))
    base = alt.Chart(coherence_df).mark_line().encode(
        x=alt.X('n_topics:Q',title = 'Number of Topics',axis=alt.Axis(labelAngle=0)),
        y=alt.Y('coherence_score:Q',title = 'Mean Coherence Score'),
        facet=alt.Facet('disaster_type:N', title=None,header=alt.Header(labelFontSize=20))
    )

    n_topics_coherence_chart = base.properties(
            title='Optimal Number of Topics (Mean Coherence Score)',
            width=600,
            height=400
        ).configure_title(
            fontSize=20,
            anchor='start',
            color='black'
        ).configure_axis(
            labelFontSize=20,
            titleFontSize=20
        ).configure_legend(
            labelFontSize=20,
            titleFontSize=20
    )
    st.write(n_topics_coherence_chart)
    st.subheader("Topic Modeling Results")

    st.markdown("**pyLDAvis**")
    """
    [pyLDAvis](https://pyldavis.readthedocs.io/en/latest/index.html) is an open source python package for interactive topic modeling visualization.
    
    pyLDAvis was a great tool for us to use as we iteratively improved our LDA models for the following key reasons: 
    1. We can see how well the topics are separated in 2D space by visualizing each topic by it's first two principal components
    2. We can see the dominant words in each topic along with how frequent those words are across the entire corpus
    3. Taking a qualitative approach, we can gauge the logical coherence of our topics by observing the top keywords 
    4. We were able to identify words that were heavily weighted across all the topics and add them to our list of stopwords
        - By excluding these "common" words across the four disaster types, we were able to improve our coherence scores as each topic became more unique   
    """
    top_sentences = pd.read_csv(os.path.join(loc.blog_data, 'lda_top_sentence.csv'))

    st.markdown("**Topic Identification and Exploration**")

    """Below we have given you the ability to select a disaster type and view the corresponding topics we've generated.
     For each disaster type, you can see the topics identified along with the most representative tweet from our 
     training data for each topic. You can also interact with the topics and keywords for each disaster type using the 
     pyLDAvis visualization.
     """

    disaster_type = st.radio(
        "Select a Disaster Type:",
        ('Earthquake (n_topics=2)', 'Fire (n_topics=4)', 'Flood (n_topics=3)', 'Hurricane (n_topics=5)'), horizontal=True)

    with open(os.path.join(loc.blog_data, 'lda_earthquake_pyldavis.html'), 'r') as f:
        earthquake_ldavis = f.read()
    with open(os.path.join(loc.blog_data, 'lda_fire_pyldavis.html'), 'r') as f:
        fire_ldavis = f.read()
    with open(os.path.join(loc.blog_data, 'lda_flood_pyldavis.html'), 'r') as f:
        flood_ldavis = f.read()
    with open(os.path.join(loc.blog_data, 'lda_hurricane_pyldavis.html'), 'r') as f:
        hurricane_ldavis = f.read()

    if disaster_type == 'Earthquake (n_topics=2)':
        st.write('You selected Earthquake which we identified as having primarily two topics:')
        st.write('**Shown below is the most representative tweet for each topic:**')
        st.table(top_sentences[top_sentences['disaster_type'] == 'earthquake'][['topic', 'tweet_text']])
        components.v1.html(earthquake_ldavis, width=1300, height=800, scrolling=False)
    elif disaster_type == 'Fire (n_topics=4)':
        st.write('You selected Fire which we identified as having primarily four topics:')
        st.write('**Shown below is the most representative tweet for each topic:**')
        st.table(top_sentences[top_sentences['disaster_type'] == 'fire'][['topic', 'tweet_text']])
        components.v1.html(fire_ldavis, width=1300, height=800, scrolling=False)
    elif disaster_type == 'Flood (n_topics=3)':
        st.write('You selected Flood which we identified as having primarily three topics:')
        st.write('**Shown below is the most representative tweet for each topic:**')
        st.table(top_sentences[top_sentences['disaster_type'] == 'flood'][['topic', 'tweet_text']])
        components.v1.html(flood_ldavis, width=1300, height=800, scrolling=False)
    elif disaster_type == 'Hurricane (n_topics=5)':
        st.write('You selected Hurricane which we identified as having primarily five topics:')
        st.write('**Shown below is the most representative tweet for each topic:**')
        st.table(top_sentences[top_sentences['disaster_type'] == 'hurricane'][['topic', 'tweet_text']])
        components.v1.html(hurricane_ldavis, width=1300, height=800, scrolling=False)

with tab_3:
    st.header(tabs_list[2])

    st.subheader("Traditional Algorithms")

    st.markdown("**Summary**")

    st.write("""We attempted two kinds of classification models on our disaster tweets dataset:""")
    
    st.write("""
1. **Classifiers that determine if a tweet pertains to a particular disaster (wildfire, flood, hurricane,earthquake, non-disaster).** 
Initially, we tried retrieving a sample of random tweets to use as a non-disaster related tweets by applying a "non-disaster" label.
We then assigned the "disaster" label (i.e. wildfire, flood, hurricane, earthquake) to the tweets from our disaster tweets training
sample. Our thought was that we would then be able to build a binary classification model off of this combined dataset
 to determine if a particular tweet was disaster related or not. While our trained classification algorithms performed 
 very well on the disaster dataset alone - (we achieved an F1 score above 0.93 for the logistic regression model), when 
 we attempted to apply this classifier to the combined labeled dataset, most of the predicted labels for non-disasters 
 were labeled as "hurricane" as opposed to "non-disaster". Therefore, we abandoned this approach and resorted to 
 leveraging the disaster specific keywords from our topic modeling to develop twitter queries.""")

    st.write("""
2. **Classifiers that compute the class label after getting trained on the disaster tweet dataset.**

We managed to achieve a macro-F1 score of 0.71 on the test dataset using multi-class classification
logistic regression. We applied the model later to a random sample tweets from the past week,
and even though we did not have labeled data there, our observation was that the predicted labels looked correct and
reasonable. The labels the algorithm learned to predict were as follow:
  - rescue_volunteering_or_donation_effort
  - other_relevant_information
  - requests_or_urgent_needs
  - injured_or_dead_people
  - infrastructure_and_utility_damage
  - sympathy_and_support
  - caution_and_advice
  - not_humanitarian
  - displaced_people_and_evacuations
  - missing_or_found_people
    """)

    st.markdown("**Preprocessing and vectorization of the dataset**")

    st.write("""The first step of the data processing is the vectorization, where we find a proper representation of each
    tweet. Once vectorized, we tried to compare the effect of each technique by trying out the respective
    representation on the logistic regression model to see which one yields the best classification result.
    The main things we attempted were as follows:
    - Using tweet tokenizer from the nltk library ([TweetTokenizer](https://www.nltk.org/api/nltk.tokenize.casual.html#nltk.tokenize.casual.TweetTokenizer))
    with TF/IDF vectorization. This produced a sparse matrix that can be used in a ML algorithm. This tokenization
    when used with logistic regression produced an F1 score of 0.714. The generated notebook is vectorizer.ipynb and can 
    be found in the project output folder
    - Using the same tokenization and vectorization as above, but adding an SVD step to project the 
    sparse matrix to a reduced dimension. The result was an F1 score of 0.681 and
    the outcome can be seen in vectorizer_svd.ipynb
    - Using [spaCy](https://spacy.io/) library with stemming, stop words removal and again TF/IDF vectorization.
    This again produced a sparse matrix and when tested with a logistic regression model the F1 score was 0.701.
    The generated notebook is vectorizer_spacy.ipynb
    - Word2Vec dense matrix vectorization with [gensim](https://github.com/RaRe-Technologies/gensim) library.
    The F1 result we achieved was 0.591 - probably one of the lowest so far. The respective Jupyter notebook 
    is vectorizer_dense.ipynb. 
    - Count vectorization which was only used for the LDA topic modeling effort. The generated notebook is called 
    vectorizer_countVec.ipynb""")

    st.write("""
    After comparing the results, we ended up using the first vectorization option for our models - tweet tokenizer
    with TF/IDF from sklearn, since it was producing the best evaluation scores.
    """)

    st.markdown("**Classification algorithms selection and comparison**")

    st.write("""
    For the class label classification, we tested the following algorithms and algorithm combinations:""")
    st.write("- Linear regression - we achieved the 0.710 average macro F1 score leveraging a grid search technique.")
    st.write("""- Random Forest - this algorithm was disappointing for this multiclass scenario with average macro F1 score of 0.207. As a comparison, the dummy classifiers were giving scores from 0.04 to 0.08.""") 
    st.write("""- Multinomial NB - the F1 score was 0.569, better than random forest, but worse as compared to the logistic regression.""")
    st.write("""- Voting classifier, where the three algorithms above were voting on the label. The averaged macro F1 score on this one came to 0.689, again, not as good as the logistic regression, but better than the others.
    We alo tried the Gradient Boosting Classifier, but we had to interrupt it, because it was taking far too long to train.""")

    st.subheader("Deep Learning Models")
    """
    To complete our model evaluation we also tested several deep learning model variations. Given the high performance on many benchmark assessments 
    that neural networks have achieved in the past several years, and especially their performance on natural language data we wanted to test several neural networks 
    on our data set. To begin, we built a fairly simple Embedding Bag model using PyTorch. An embedding bag can be thought of as a two step process (although this is 
    not implemented as a two step process in PyTorch). First, all the sentences in a batch are combined into one long tensor and the tensor is embedded into some 
    n-dimensional space, also known as the embedding dimension, which is chosen by the user. Next, a reduction (mean, min, max, etc.) is applied across the embedded 
    latent dimension and this is passed to a fully connected layer which then produces the predictions. This neural architecture is similar to what would happen if you 
    vectorized text via one of the many methods discussed throughout the UMADS program, word2vec, tf-idf, and others, then passed this output into a fully-connected 
    neural network. It is simple but very fast, and achieved performance similar to that of our standard logistic regression.

    However, one major weakness of the embedding algorithm is that it does not consider the order of the words or structure of the sentence - all words are embedded simultaneously. 
    We felt it would be valuable to also evaluate a more sophisticated model that makes some of these considerations. For this we chose to incorporate a Long-Short Term Memory (LSTM) neural network. 
    The architecture was similar to above; the network included an embedding layer (although not an embedding bag - the sentence vectors were not reduced after the embedding) followed by a LSTM 
    followed by several fully connected layers. We hoped that the LSTM and its ability to consider a tweet as an entity, not just a collection of words, would improve our classification accuracy. 
    Unfortunately this was not the case. Even after hyperparameter tuning and several iterations the model architecture (ex: number of layers, size of embedding and hidden dims, bidirectionality, and more) 
    we were not able to surpass the accuracy of the simpler logistic regression and embedding bag models. After many epochs we also noticed that the model would begin to overfit and perform very well on the 
    training data while not gaining the same accuracy increases on the validation set.

    While it is possible that with more finetuning we would have been able to make slight increases in the model accuracy, we felt the model was limited by the data in 
    several ways. First, there is such a wide variation in the structure of the texts, sometimes the words are just as important as the structure, which means that using a LSTM will not significantly increase the 
    accuracy of the model Additionally, we were working with a fairly small data set, only about 50k records, which impacts the ability of model to learn the data.
    """

    st.subheader("Final Observations and Conclusions")
    """
    In the end we decided to go with a simpler logistic regression model because it is easier to build, train, and maintain than the other deep learning models we tested while performing on par with these models.
    """

with tab_4:
    st.header(tabs_list[3])
    st.markdown("**Summary**")
    st.write(
    """The objective of training tweet classification models for the purpose of this project is to use them later
    on recent tweets collected via the twitter api, simulating a real-time application. Our objective is to answer the 
    following questions: """)
    """
    * Is there an active disaster going on?

    * What kind of disaster is going on - wildfire, earthquake, flood, hurricane?

    * What are the affected locations, are some locations connected together?

    * For the active disasters, what categories do the tweets fall in - distress tweets, ask for donations etc.?    

    * What are the recommended actions in regard to the disaster?

    * What is the disaster intensity over time, what are the peak days?

    * Can we preview the tweets for a particular time to better understand what is going on?"""

    """
    The following section explains how approached answering these questions.

    For the purpose of this demonstration we have obtained and analyzed tweets for the week between July 31st, 2022
    and August 7th, 2022.
    """
    st.subheader("Obtaining sample tweets by querying the Twitter APIs")
    """
    The first step in analyzing the most resent disaster tweets is obtaining them from Twitter. The solution we
    settled on to achieve that is as follows:"""
    """
    * Using the keyword tokens obtained from the unsupervised topic exploration of the disaster sample, we query the Twitter API for the past six days (this is max time allowed to go back) and for the tweets up to certain count (default set to 2000) for every two-hour period. An example of the queries we are using for the respective disasters are:"""
    disaster_table = pd.DataFrame({"Disaster":["Wildfire", "Earthquake", "Flood", "Hurricane"],
                            "Twitter Query": ["wildfire (donate OR (death toll) OR burn OR fire OR forest OR damage)",
                                              "earthquake (pray OR victim OR donate OR rescue OR damage OR magnitude)",
                                              "flood (relief OR water OR donate OR (death toll) OR rescue OR destroy)",
                                              "hurricane (donate OR storm OR cyclone OR damage OR evacuation OR destroy)"]})
    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    st.table(disaster_table)

    """And the number of tweets retrieved is about 144,000 per disaster, or about 576,000 total. The details of
    the retrieval process are in the `recent_tweets_*.ipynb` notebooks in the project output folder."""

    st.subheader("Assigning a class label on the retrieved disaster tweets")
    """The obtained tweets are labeled with one of following categories using the vectorizer and the classification algorithm prepared in the upstream steps:"""

    """
    * rescue_volunteering_or_donation_effort
    * other_relevant_information
    * requests_or_urgent_needs
    * injured_or_dead_people
    * infrastructure_and_utility_damage
    * sympathy_and_support
    * caution_and_advice
    * not_humanitarian
    * displaced_people_and_evacuations
    * missing_or_found_people
    The completed dataset is ready for further analysis.
    """
    st.subheader("Determining the dominating disaster in a category")
    """
    The first question we are trying to answer on the newly obtained dataset is find out if there is an
            active disaster, and what kind it is. For that purpose, we group all the tweets per disaster type and 
            class label, and find the count of the tweets for 8-hour intervals. A sample of the result 
            dataframe looks like this:"""

    st.image(os.path.join(loc.blog_data, "grouped_tweets.png"), caption=None)

    """
    When we create a boxplot for the tweet frequency distribution by disaster type, we can conclude
            that there is no visible difference in the mean and variance of the samples:"""

    st.image(os.path.join(loc.blog_data, "class_distribution.png"), caption=None)

    """
    We can statistically prove the same by running an f-test on the disaster distributions. The 
    result is:
    ```buildoutcfg
    F_onewayResult(statistic=0.3869087889615666, pvalue=0.7624738994702899)
    ```
    """
    """
    The f-statistic is low and below 10, the p-value is high for the 10% significance, therefore,
    we cannot reject the null hypothesis that the mean frequency of the individual disaster samples are the same.

    How about if we filter the tweets for the 'displaced_people_and_evacuations' class label. In
    that case the boxplot chart looks like that:
    """
    st.image(os.path.join(loc.blog_data, "displaced_class_distribution.png"), caption=None)
    """
    For this particular class the mean and the variance of the wildfires are visibly higher as
    compared to the others, which makes sence since there were many active fires in California
    at that time. We can run an f-test and observe the result:
    ```buildoutcfg
    F_onewayResult(statistic=6.998888294867653, pvalue=0.00043411742534776675)
    ```
    The p-value is low, which indicates that there are differences in the mean and variance,
    however, the f-statistic is not that high - ideally it should be above 10 for us to 
    confidently reject the null hypothesis.

    We can run also a one-sided KS test for each disaster type vs all the others and determine
    if one clearly dominates. The function that we have written to compute that returns the
    following:
    ```buildoutcfg
    dominatingDisaster=None, pvalues=None
    ```
    Which points to the fact it is unable to find a disaster where the number tweets in one
    category clearly exceed the others. 

    If we do a comparative visualization using again boxplot and filter by 'rescue_volunteering_or_donation_effort',
    we get the following chart:
    """
    st.image(os.path.join(loc.blog_data, "domation_class_distribution.png"), caption=None)
    """
    Which clearly shows the tweets related to flood are clearly a majority (this week was after devastating
    floods in Kentucky). We can prove this statistically running again an f-test:
    ```buildoutcfg
    F_onewayResult(statistic=63.466941273877204, pvalue=1.1924304959724865e-20)
    ```
    The p-value is very low, the f-statistic is well above 10, which means the mean and variances of the
    observed groups are clearly different.

    Utilizing the paired KS-test function we have written, we can point exactly which disaster kind is the
    majority - it is "Flood" and the one-sided test p-values are proof of that:
    ```buildoutcfg
    dominatingDisaster=Flood, pvalues=[3.396884814555207e-08, 5.429149940086322e-08, 3.046866711808434e-09]
    ```

    How can we determine if there is an active disaster going on for this particular time? Our solution
    was to look for class labels that were indicating more distress tweets, in particular, we defined 
    an "active group" of tweets with the labels 'displaced_people_and_evacuations' and 'injured_or_dead_people':
    ```buildoutcfg
    class_labels_active = [
        'displaced_people_and_evacuations',
        'injured_or_dead_people',
    ]
    ```
    When using both distress classes, the visualization chart is:
    """
    st.image(os.path.join(loc.blog_data, "active_class_distribution.png"), caption=None)
    """
    The respective f-test statistic is:
    ```buildoutcfg
    F_onewayResult(statistic=10.35071997380027, pvalue=3.960570565233521e-06)
    ```
    And the KS-test clearly identifies the "Wildfire' as a dominant active disaster just as expected:
    ```buildoutcfg
    dominatingDisaster=Wildfire, pvalues=[0.07322472520068853, 5.2459936908087666e-05, 2.3038948480462642e-07]
    ```
    Similarly, we can identify classes that are mostly post-active disasters - like the floods are gone,
    however, the community still deals with the consequences. In this category we included the following
    classes:
    ```buildoutcfg
    class_labels_past = [
        'rescue_volunteering_or_donation_effort',
        'requests_or_urgent_needs',
        'sympathy_and_support',
    ]
    ```
    The visualization again shows the "Flood" as dominant, and the statistical tests emphasize that:
    """
    st.image(os.path.join(loc.blog_data, "past_class_distribution.png"), caption=None)
    """
    The f-test result as follows - low p-value and high f-statistic, the means and variances are clearly
    different:
    ```buildoutcfg
    F_onewayResult(statistic=11.950317300994135, pvalue=3.715889348149974e-07)
    ```
    And the KS-test points out to the "Flood":
    ```buildoutcfg
    dominatingDisaster=Flood, pvalues=[0.007012714471607445, 0.014540857905105333, 0.006384398236281075]
    ```
    The class "caution_and_advice" can be used as an indication for upcoming/expected disasters,
    in our exploration, the "Hurricane" came on top in this category.

    We created some additional visualizations that show the tweet intensity over time, and potentially
    can point out at what time a particular category was truly active:
    """
    st.image(os.path.join(loc.blog_data, "past_class_timeline.png"), caption=None)

    st.subheader("Final Observations and Conclusions")
    """
    We demonstrated how the recent tweets about a particular disaster category can be obtained and 
    analysed both visually and by leveraging statistical tests in order to identify dominant
    active, past and possibly future disasters. We could not accomplish this without the ML
    classification models we trained on the labeled dataset. 

    In the next section we will show how we extract the locations of the affected places, and how
    we leverage this info to point out more details and insights for a particular disaster. 
    """

with tab_5:
    st.header(tabs_list[4])

    st.subheader("Summary")
    """
    The comparative analysis section explained how the recent disaster tweets were queried and retrieved,
    and then processed for finding a dominant disaster type per class. The following section continues the 
    exploration by doing a location extraction and interpretation exercise, along with showing creative ways to
    sample and visualize tweets based on a specified criteria. The exploration is using the [Altair](https://altair-viz.github.io/) 
    library, which provides interactive features, however, the charts are not visible directly in GitHub.
    For that reason, the produced Jupyter notebooks are exported as PDF files in folder docs/pdfs. Of course,
    you can run the project yourself and play with the generated interactive artifacts.

    These are the key questions which will be addressed:
    * What are the affected locations, are some locations connected together?
    * What is the disaster intensity over time, what are the peak days?
    * Can we preview the tweets for a particular time and understand better what is going on?
    """
    st.subheader("Tweet Location Extraction")
    """
    For the purpose of this disaster analysis, the locations are extracted from the tweet text and not
    from the associated metadata attributes. The rational for this is that a relevant tweet is
    not necessarily posted at the specific disaster location.

    We are using [spaCy](https://spacy.io/) library with the xx_ent_wiki_sm pipeline, which is good for
    extracting locations from text.

    Once the location extraction process is complete, a new column is added to the recent tweets data that contains
    a list of all locations extracted from the tweet. Therefore, there may be two or more locations in
    the same text:
    """
    st.image(os.path.join(loc.blog_data, "tweets_locations_table.png"), caption=None)
    """
    The top locations by tweet count are then identified and ready to be used. For the "wildfire" 
    disaster type for example, for the period between July 31st, 2022 and August 7th, 2022, the top
    fire locations are:
    """
    st.image(os.path.join(loc.blog_data, "top_locations.png"), caption=None)

    st.subheader("Disaster tweets interactive exploration")
    """
    The first thing is to do with the updated data is to create an interactive chart that to show us
    what are the affected locations over time. As expected, California is very active, followed by Utah (brown line):
    """
    st.image(os.path.join(loc.blog_data, "fire_locations.png"), caption=None)

    """
    Running the classification models assigns a class label on each tweet in the dataset. The respective
    interactive visualization now shows what class and locations are affected at a particular time,
    and allows us to explore the peaks:
    """
    st.image(os.path.join(loc.blog_data, "class_locations_peaks.png"), caption=None)
    """
    A subsequent interactive visualization allows us to quickly sample the tweets for a selected time period
    and gives an even better idea of what is going on with the particular disaster at that time:
    """
    st.image(os.path.join(loc.blog_data, "fire_tweet_sample.png"), caption=None)
    """
    The fact that we collect multiple locations from a tweet allows us to do further exploration and see
    what locations are used together. This allows us to quickly understand and narrow down some less
    known locations - i.e. in the following interactive chart for the class "Caution And Advice", we can
    see that the primary location is California and Northern California. However, the exact location of
    the fire is Klamath National Forest:
    """
    st.image(os.path.join(loc.blog_data, "location_caution_network.png"), caption=None)
    """
    The class "Injured or dead people" interactive location network visualization is even more interesting,
    because is shows more connections - we can see that the fires in California are likely
    affecting the neighbouring state of Oregon. Additionally, we can see that Klamath National Forest is in California, along with the Siskiyou County 
    in northern California etc:
    """
    st.image(os.path.join(loc.blog_data, "location_injured_network.png"), caption=None)
    """
    The next visualization shows two completely different locations related to two separate flood
    disasters - one is in the state of Kentucky, the other one is in Pakistan - we can see that two 
    separate networks are getting formed, and we can explore each one of them. We also can see that 
    the location extraction is not perfect - i.e. we have the word "Helicopter" as a location, most probably
    because in the tweet is starts with a capital letter and was mistakenly confused for a proper noun:
    """
    st.image(os.path.join(loc.blog_data, "location_flood_rescue.png"), caption=None)

    """
    The exploration also allows us to randomly sample tweets for multiple overlapping locations and understand
    better how are they connected.
    """

    st.subheader("Final Observations and Conclusions")
    """
    Interactive visualization and network diagrams can be a powerful tool to provide a deeper understanding of the currently active natural
    disasters going on in the world, providing insight into the affected locations and the various disaster response subcategories. 
    """

with tab_6:
    st.header(tabs_list[5])
    
    st.subheader("Tweet Classication and Filtering")
    
    st.write("""
    As discussed in the project introduction, information is extremely hard to come by during a natural disaster.  Depending on the type and scope of the disaster phone lines and other traditional means of communication may be disbled, making information via the internet, and specifically Twitter, extremely valuable.  However, Twitter is a massive ecosystem and it can be very difficult to use the correct search terms when looking for information, and equally difficult to find useful information within the massive amount of results even when proper search terms are used.  
    
    Our project would be invaluable to those in the midst of a natural disaster by aggregating Tweets, classifying them, and making them easier to search - in essence we built a natural disaster search engine on top to Twitter.  The information and resources revealed via our search tool could range from sites where donations were being collected to locations where resources and food could be picked up by those affected.  Below is a small demonstration of how our tool works: first, Tweets relevant to natural disasters - collected via our disaster specific search term topics and classified as a relevant Tweet via our classification model - are parsed, cleaned and stored with additional data points like `Geo-Political Entity` extracted and saved.  Next, a second classification model predicts which one of ten classes the Tweet belongs to, adding another layer of searchability.  
    
    Finally the Tweets are aggregated and sorted by re-Tweet count resulting in collection of relevant, curated, searchable, natural disaster specific Tweets.
    """)
    
    tweet_data = pd.read_csv(os.path.join(loc.blog_data, "classification_data_sample.csv"))
    with open(os.path.join(loc.blog_data, "locations"), "rb") as l:
        locations = pickle.load(l)
    
    tweet_categories = ['rescue_volunteering_or_donation_effort',
       'other_relevant_information', 'requests_or_urgent_needs',
       'injured_or_dead_people', 'infrastructure_and_utility_damage',
       'sympathy_and_support', 'caution_and_advice', 'not_humanitarian',
       'displaced_people_and_evacuations', 'missing_or_found_people']

    tweet_types = st.multiselect("Choose a tweet topic", tweet_categories, tweet_categories[:])

    start_date = st.date_input(
        "Select Minimum Tweet Date",
        datetime.date(2020, 1, 1),
        min_value=datetime.datetime.strptime("2020-01-01", "%Y-%m-%d"),
        max_value=datetime.datetime.now(),
    )

    date_filter = pd.to_datetime(tweet_data.created_at).apply(lambda x: x.date()) >= start_date

    tweet_data_filtered = tweet_data[(tweet_data.predicted_class.isin(tweet_types)) & (date_filter)]

    tweet_data_filtered["created_at"] = pd.to_datetime(tweet_data_filtered.created_at).apply(lambda x: x.date())

    location_list = st.multiselect("Specify a location", locations, ["all"])

    # tweet_data_filtered = tweet_data_filtered[tweet_data_filtered.locations.isin(location_list)]
    
    tweet_data_filtered = tweet_data_filtered[tweet_data_filtered['locations'].apply(lambda a: check_locs(a, location_list))]

    tweet_data_filtered = tweet_data_filtered[["created_at", "tweet_text", "name", "predicted_class", "tweet_count"]].copy()

    st.write(tweet_data_filtered.head())
    
    st.subheader("Actionable Insights")
    
    """
    In addition to curating and aggregating useful Tweets our project also extracts useful information from the Tweets - specifically those Tweets that suggest or recommend some concrete action - and makes it available in a easy to consume format.  Again, in the information overload that is Twitter it can be hard to easily find resources to access (for those involved in the disaster), or how/where to make donations and help (for those not directly involved in the disaster).  
    
    To accomplish this we developed a three step process.  First, we queried current Twitter for natural disaster related information by 
    searching for the most relevant topics and words for each disaster type, as determined by our topic models.  Next, we applied our classification model and kept only the Tweets that were labeled as `rescue_volunteering_or_donation_effort`, hypothesizing that these Tweets would be the most important for those searching for resources or with resources to donate.  At this step we also deduplicated re-Tweets and kept a track of the count, using this re-Tweet count as our sorting mechanism (most re-Tweeted to least re-Tweeted.  Finally, we used Spacy to extract the important information from the Tweet in order to concisely display what action the Tweet was recommending and where the user could go (via hyperlink) to accomplish the goal.
    
    Below we demonstrate how we have extracted the key information from the Tweet while also preserving the original Tweet.  We believe that by making it easy to access resources or contribute to a recovery effort we can lower the barrier to entry and get more people the help they need while simealtaneously increasing the number of people donation, volunteering or aiding in some other fashion.
    """
    
    v_spacer(height=3)
    
    # nlp = spacy.load("en_core_web_sm")
    # stopwords = nlp.Defaults.stop_words
    col1, col2 = st.columns(2)

    with col1:
        start_date_action_tweets = st.date_input(
            "Select Minimum Tweet Date For Actions",
            datetime.date(2020, 1, 1),
            min_value=datetime.datetime.strptime("2020-01-01", "%Y-%m-%d"),
            max_value=datetime.datetime.now(),
        )

    with col2:
        verb_list = ["donate", "volunteer", "all"]

        verb_types = st.multiselect("Choose a tweet topic", verb_list, verb_list[:])


    # load data
    # use .pkl not .csv because .csv does not preserve the Spacy text
    action_tweets = pd.read_pickle(os.path.join(loc.blog_data, "action_tweets_data_sample.pkl"))
    
    # create filter
    date_filter_action_tweets = pd.to_datetime(action_tweets.created_at).apply(lambda x: x.date()) >= start_date_action_tweets
    
    # apply filter
    action_tweets_filtered = action_tweets[date_filter_action_tweets]
    
    regex = re.compile('|'.join(re.escape(x) for x in verb_list), re.IGNORECASE)

    for idx, data in action_tweets_filtered.head(20).iterrows():
        # find the recommended action
        verb_matches = re.findall(regex, data["tweet_text"])
        if len(set(verb_matches).intersection(set(verb_types))) < 1:
            continue
        total_tweet_count = data.tweet_count - 1

        # at least on word has been found
        if len(verb_matches) > 0:

            verb_matches.append("all")
            # find all the links (often more than 1 donation site)
            donation_url_list = []

            # check for a retweet
            original_tweeter = re.findall("RT @([a-zA-z0-9_]*)", data["tweet_text"])

            # find and record all the urls in the tweet
            for token in data["spacy_text"]:
                if token.like_url:
                    donation_url_list.append(token)


            if len(donation_url_list) > 0:
                if len(original_tweeter) > 0:
                    tweet_author = original_tweeter[0]
                else:
                    tweet_author = data["name"]
                for idx, url in enumerate(donation_url_list):
                    if idx == 0:
                        st.write(f"{tweet_author} and {total_tweet_count} others recommend you {make_str(verb_matches)}.  More information at {url}")
                    else:
                        st.write(f"Please also consider donating to {url}")
                with st.expander("Original Tweet"):
                    st.write(data['tweet_text'])
                st.write("\n\n")
                st.write("-"*50)

with tab_7:
    st.header(tabs_list[6])

    st.subheader("Conclusions and Future Evolution of the System")

    """As part of this disaster tweets analysis project we explored various ML models
    capable of predicting class labels of individual tweets, determining that the Logistic Regression model did just as good
     if not better than the the Neural Network model for this multiclass classification problem. Furthermore,
    we performed an unsupervised topic modeling analysis and extracted topic keywords that we used to
    query recent disaster tweets. With these collected tweets we then simulated a real time analysis, by identifying
    which disaster is dominant for a particular timeframe along with identifying the current active disasters. In addition,
    we extracted the locations from the sample of recent tweets and provided various interactive
    visualizations showing the relationships between different locations. Lastly, we demonstrated that we can extract 
    the recommended actions from a corpus of disaster tweets. In essence, we built a system
    that can filter the twitter fire house and present an accurate picture of historical and current natural disasters 
    occuring in the world, along with providing actionable information to those in need of timely and reliable 
    information.
    """
    """
    Some possible ways this system can evolve in the future: 
    * **Building a batch or a near-realtime system** that pulls the tweets from the past hour and analyses them, 
    publishing a report with the result. This will allow someone to follow up daily of what is going on in the world.
    * **Collecting and interpreting the disaster information over longer period of time** - like months and years. 
    Unfortunately Twitter limits how far back in time you can go through the APIs and obtain the relevant info. 
    Therefore, some additional storage needs to be added, so the long-term data is available for additional analysis and insights. 
    * **Doing a realtime alerting** when some truly disastrous/important event happens with location details and other pertinent info
    * **Adding a multilingual support** - presently the system interprets tweets just in English, and as a result it 
    may be missing important information for locations and countries that use different languages.
    * **Extracting other Twitter information, not just for disasters**, but for other scenarios: geo-political, stock 
    market, energy prices, government actions etc. 
    * **Interpreting information not just from tweets, but other news sources** - theoretically other NLP sources are 
    not that different as compared to the tweets. So they can be used as a complimentary or a primary source of the required information.
    """

with tab_8:
    st.header(tabs_list[7])

    st.subheader("Data & Resources")
    """
    * [Project Code on Github](https://github.com/Data-Medics/umads_697_data_medics)
    * Labeled Twitter Data
        * [CrisisNLP](https://crisisnlp.qcri.org/humaid_dataset.html#)
    * Leveraged Python Libraries
        * [Tweepy](https://www.tweepy.org/)
        * [Spacy](https://spacy.io/)
        * [PyLDA](https://pyldavis.readthedocs.io/en/latest/readme.html)
        * [PyTorch](https://pytorch.org/)
        * [Pandas](https://pandas.pydata.org/)
        * [Sklearn](https://scikit-learn.org/)
    * This app was built with [Streamlit](https://streamlit.io/)

    """
