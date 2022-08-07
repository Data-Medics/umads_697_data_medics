# Unsupervised Learning / Topic Modeling

## Summary
One of the primary objectives for this project is to be able to provide a list of 
recommended actions related to a particular disaster. In order to leverage this solution we needed to
acquire and identify a collection of current tweets related to a current natural disaster. We developed a function to extract
tweets from the Twitter API based on a query of keywords. The keywords we used to build this query were a direct
product of the Latent Dirichlet Allocation (LDA) topic modeling analysis we had conducted on the disaster tweets dataset.

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


For our text vectorization, we needed to convert our lemmatized text into a bag-of-words representation which the LDA model
could use. We used the sklearn [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) 
class to produce a sparse representation of token counts. The CountVectorizer class conveniently let us set our ngram_range to (1,2) along 
with filtering out our pre-defined stopwords.
 
#### Model Selection and Training
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

#### Hyperparameter Tuning
The final step before deploying the LDA model was to estimate the optimal number of topics for each disaster type LDA model. We developed a script
that performs a grid search exersice, varying the number of topics from 2-6 across each disaster type LDA model. We then computed the
log-liklihood and coherence scores for each configuration. When calculating coherence, we used the 'u_mass' measurement as opposed to
the 'c_v' measurement. The 'c_v' measurement is known to produce the most reliable results, but opted to use the 'u_mass' measurement
to speed up the computation. The full grid search took approximately XXX hrs to run. See figure XXX below for the results.

#### Topic Modeling Results