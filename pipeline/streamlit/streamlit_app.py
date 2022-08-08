import streamlit as st
import pandas as pd
#%%

#%%
def get_random_disaster_tweets():
    result = st.button('Get Disaster Tweets')
    if result:
        gather_tweets = st.text('Gathering Tweets')
        random_tweets = pd.read_csv('../../data/HumAID_data_v1.0/all_combined/all_train.tsv', sep='\t').sample(20000)
        return st.write(len(random_tweets)), st.write(random_tweets.head()), gather_tweets.success('Done Gathering Tweets')

def main():
    st.set_page_config(layout="wide")
    st.title('DATA MEDICS')
    "rjgillin@umich.edu"
    st.header('Disaster Tweet Pipeline')
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary", "Get Data", "Text Preprocessing", "Classification", "Recommended Actions"])
    with tab1:
        st.header("Project Summary")
        "Add flow diagram here and overall project explanation"

    with tab2:
        "Click button to randomly select a sample of disaster 20k disaster tweets"
        st.subheader('Get Disaster Tweets')
        get_random_disaster_tweets()

    with tab3:
        st.header("Text Preprocessing")
        st.subheader('Clean and Vectorize Tweets')
        "Text here explaining our text-preprocessing and vectorization steps"

    with tab4:
        st.header("Classification")
        st.subheader('Predict Disaster Tweet Class Labels')
        "Text here explaining the logistic regression model"
        st.subheader('Predict Disaster Type')
        "Text here explaining our disaster type prediction model"

    with tab5:
        st.header("Recommended Actions")
        'Extract the actions, recommendations, organizations, destinations, etc'

if __name__ == '__main__':
    main()

#%%
