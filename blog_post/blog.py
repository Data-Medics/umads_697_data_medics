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

sys.path.insert(0, "./pipeline")
import locations as loc


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
tabs_list = ["Project Summary", "Topic Modeling", "Tweet Classification", "Tweet Visualizations", "Real Time Tweet Analysis", "Additional Information"]
tab_1, tab_2, tab_3, tab_4, tab_5, tab_6 = st.tabs(tabs_list)
with tab_1:
    st.header(tabs_list[0])
    st.subheader("Project Overview")
    """Quality information is incredibly difficult to obtain following a major natural disaster and finding particular information can make a huge difference to those affected.  Lots of pertinent information is posted on Twitter following natural disasters, but it can be extremely challenging to sift through hundreds of thousands of tweets for the desired information.  To help solve this problem we built a one-stop-natural-disaster-information-shop.  Powered by our machine learning models, we make it easy to filter tweets into specific categories based on the information the user is trying to find.  We use natural language processing and topic models to search Twitter for tweets specific to different types of natural disasters.  From here we provide a wide variety of carefully curated, useful information ranging from geo-spatial analysis to understand the exact locations and severity of the natural disaster to parsing the Twitterverse for actionable behaviors.  The period following a natural disaster is incredibly difficult and dangerous for all those involved - our tool aims to make the recovery process easier and more efficient with better outcomes for those affected."""
    
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
    However, giving people critical information about a disaster that happened several years ago is of limited value.  For our project to truly help, we needed to apply our models and analysis to live Twitter data around current natural disasters.  To do this we utilized [Tweepy](https://www.tweepy.org/) and wrote several functions that pulled real time Twitter based on our disaster specific search terms which we then used as inputs to our model.
    """
    st.subheader("Project Plan")
    """
    Our project structure uses the following form:
    PLACEHOLDER FOR IMAGE
    """
    st.subheader("Project Layout")
    """
    The following tabs will walk through our work, starting with raw Tweet data all the way to producing actionable recommendations for current natural disasters.
    """
    st.subheader("Common Natural Disasters")
    st.image(os.path.join(loc.blog_data, "disasters_image.png"), caption="Disasters")
    
    
with tab_2:
    st.header(tabs_list[1])
    st.subheader("Overview")
    """
        In order to provide actionable recommendations during times of disaster, we needed a way to efficiently query recent and relevant disaster tweets from twitter.
        To do so we developed a function to extract tweets from the Twitter API based on a query of keywords. The keywords we used to build this query were a direct
    product of the Latent Dirichlet Allocation (LDA) topic modeling analysis we conducted on the HumAid disaster tweets dataset.
    """
    st.subheader("Topic Modeling Exploratory Analysis")
    st.markdown("**Token Exploration**")
    """
        To gain some initial familiarity with the disaster tweet text, we performed some simple token exploration.
    More specifically we looked at how many tokens were composed of word characters versus tokens comprised of non-word characters.
    Tweet text can be very messy, and we would eventually need to decide how to handle hashtags, symbols, retweets and mentions.
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
        Lemmatization is the process of reducing various inflectional forms of a word to it's base form. We chose lemmatization
    over stemming because we wanted to retain the morphological structure of words, believing this would be
    a more robust way to capture topics. Stemming is a more crude approach in which the end or the beginning of a word is cut
    off to reduce the word to it's root. 
    
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
    the topics we were generating were suboptimal. More specifically, the topics were blended across the various disaster types leading to poor
    interpretability. To address this, we decided to train a separate topic model for each disaster type. While this introduced additional
    algorithmic complexity, it turned out to be the right approach, as our topics became much more concise and easily interpretable. 
    
    The single key parameter for the LDA model is the "n_components" parameter. After some initial experimentation,
    it seemed that an n_components value between 2-4 was returning relatively concise topics. Eventually we would perform a more
    robust hyperparameter tuning exercise to systematically estimate the optimal number of topics, but initially we set n_components = 3,
    so we could continue with our development work. From here, we fine-tuned various aspects of our topic modeling pipeline using log-liklihood
    scores, perplexity scores and general topic interpretability to guide our refinements to the model. 
    """
    st.markdown("**Hyperparameter Tuning**")
    """  
        The final step before deploying the LDA model was to estimate the optimal number of topics for each disaster type LDA model. We developed a script
    that performs a grid search exersice, varying the number of topics from 2-6 across each disaster type LDA model. We then computed the
    log-liklihood and coherence scores for each configuration. When calculating coherence, we used the 'u_mass' measurement as opposed to
    the 'c_v' measurement. The 'c_v' measurement is known to produce the most reliable results, but opted to use the 'u_mass' measurement
    to speed up the computation. The full grid search took approximately 4 hrs to run. See figure A below for the results.
    """
    st.image("blog_post/blog_data/topic_model_gridsearch_results.png", caption="")
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
    st.subheader('Topic Identification and Exploration')
    st.write('Below we have given you the ability to select a disaster type and view the corresponding topics. For each disaster type, you can see the topics identified along '
             'with the most representative tweet from our training data for each topic. You can also interact with the topics and keywords for each disaster type using the pyLDAvis visualization.')
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

with tab_4:
    st.header(tabs_list[3])

with tab_5:
    st.header(tabs_list[4])
    
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
    
    nlp = spacy.load("en_core_web_sm")
    stopwords = nlp.Defaults.stop_words
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
    action_tweets = pd.read_csv(os.path.join(loc.blog_data, "action_tweets_data_sample.csv"))
    
    # create filter
    date_filter_action_tweets = pd.to_datetime(action_tweets.created_at).apply(lambda x: x.date()) >= start_date_action_tweets
    
    # apply filter
    action_tweets_filtered = action_tweets[date_filter_action_tweets]
    
    # add spacy nlp
    action_tweets_filtered["spacy_text"] = action_tweets_filtered["spacy_text"].apply(nlp)
    
    regex = re.compile('|'.join(re.escape(x) for x in verb_list), re.IGNORECASE)

    for idx, data in action_tweets_filtered.head(10).iterrows():
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

with tab_6:
    st.header(tabs_list[5])

