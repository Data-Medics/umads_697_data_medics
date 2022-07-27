import streamlit as st
import pandas as pd
#%%

#%%
def get_random_disaster_tweets():
    result = st.button('Get Disaster Tweets')
    if result:
        st.write('Gathering Tweets')
        random_tweets = pd.read_csv('../../data/HumAID_data_v1.0/all_combined/all_train.tsv', sep='\t').sample(frac = .2)
        return st.write(len(random_tweets)), st.table(random_tweets.head())

def main():
    st.title('Disaster Tweet Pipeline')
    get_random_disaster_tweets()


if __name__ == '__main__':
    main()
