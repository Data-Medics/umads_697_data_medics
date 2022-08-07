# Unsupervised Learning / Topic Modeling

## Summary
One of the primary objectives for this project is to be able to provide a list of 
recommended actions related to a particular disaster. In order to leverage this solution we needed to
acquire and identify a collection of current tweets related to a current natural disaster. We developed a function to extract
tweets from the Twitter API based on a query of keywords. The keywords we used to build this query were a direct
product of the topic modeling analysis we had conducted on the disaster tweets dataset.

### Topic Modeling Exploratory Analysis

#### Token Exploration
To gain some initial familiarity with the disaster tweet text, we performed some simple token exploration.
More specifically we looked at how many tokens were composed of word characters versus tokens comprised of non-word characters.
Tweet text can be very messy, and we would eventually need to decide how to handle hashtags, symbols, retweets and mentions.

### Model Design

#### Text Pre-Processing
After a few initial iterations, it became clear that the success and performance of our topic model
would largely depend on how well we were able to pre-process the disaster tweet text. This naturally became an iterative process
as we adjusted the following parameters and inputs, measuring topic quality at each iteration:
1. Stopword removal
2. Tokenization technique (single tokens, bigrams, trigams)
3. Splitting of compound words
4. Inclusion and exclusion of hashtags, twitter handles (@user), web addresses, non-ascii characters
5. Word length


#### Lemmatization & Vectorization
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



#### Model Selection and Training
We chose to build an LDA (Latent Dirichlet Allocation) model to extract topics from our disaster tweets. We used the
sklearn implementation for this project. When choosing a text vectorization method, we were limited to a "bag of words"
representation since that is the required data format for an LDA model.    

#### Hyperparameter Tuning


#### Topic Modeling Results