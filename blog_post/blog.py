import streamlit as st
import os
import pandas as pd
import sys

sys.path.insert(0, "./pipeline")
import locations as loc



st.set_page_config(layout="wide")
st.title('DATA MEDICS')
st.header('Disaster Tweet Pipeline')
tabs_list = ["Project Summary", "Topic Modeling", "Tweet Classification", "Tweet Visualizations", "Actionable Insights", "Additional Information"]
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
    However, giving people critical information about a disaster that happened several years ago is of limited value.  For our project to truly help, we needed to apply our models and analysis to live Twitter data around current natural disasters.  To do this we utilized [Tweepy](https://www.tweepy.org/) and wrote several functions that pulled real time Twitter based on our disaste specific search terms which we then used as inputs to our model.
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
    
with tab_2:
    st.header(tabs_list[1])

with tab_3:
    st.header(tabs_list[2])

with tab_4:
    st.header(tabs_list[3])

with tab_5:
    st.header(tabs_list[4])
    
with tab_6:
    st.header(tabs_list[5])

    