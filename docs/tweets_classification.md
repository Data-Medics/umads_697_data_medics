# Tweets Supervised Learning and Class Label Classification

## Summary
We attempted two kind of classification models on our disaster tweets dataset:
- __Classifiers that determines if a tweet is a particular disaster (wildfire, flood, hurricane,
earthquake, non-disaster)__. We tried retrieving a random tweets sample, and used that as a 
"non-disaster" label, and then assigned the "disaster" labels to the tweets from the disaster
sample. The trained classification algorithms worked very well on the disaster dataset - 
F1 score above 0.93 for the logistic regression one. However, when we attempted to apply this to a
random tweeter sample  most of the predicted labels were coming as "hurricane". Therefore,
we did abandon the idea and resorted to leveraging the tokens from the topic modeling combined
with a creative twitter search queries.


- __Classifiers that computes the class label after getting trained on the tweet disaster dataset.__
We managed to achieve a macro-F1 score of 0.71 on the test dataset using multi-class classification
logistic regression. We applied the algorithm later on ransom sample tweets from the past week,
and even we did not have labels there, our observation was that the assigned labels look correct and
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

## Preprocessing and vectorization of the dataset
The first step of the data processing is the vectorization, where we find a proper representation of each
tweet. Once vectorized, we tried to compare the effect of each technique by trying out the respective
representation on logistic regression model, to see which one yields the best classification result.
The main things we attempted were as follows:
- Using tweet tokenizer from the nltk library ([TweetTokenizer](https://www.nltk.org/api/nltk.tokenize.casual.html#nltk.tokenize.casual.TweetTokenizer))
with TF/IDF vectorization. This produced a sparse matrix that can be used in a ML algorithm. This tokenization
when used with logistic regression produced an F1 score of 0.714. The generated notebook is vectorizer.ipynb and can be found in the project output folder
- Using the same tokenization and vectorization as above, just with adding an SVD step projecting the the 
sparse matrix in a lower dimension to produce a more dense matrix. The result was an F1 score of 0.681 and
the outcome can be seen in vectorizer_svd.ipynb
- Using [spaCy](https://spacy.io/) library with stemming, stop words removal and again TF/IDF vectorization.
This again produced a sparse matrix and when tested with a logistic regression model the F1 score was 0.701.
The generated notebook is vectorizer_spacy.ipynb
- Word2Vec dense matrix vectorization with [gensim](https://github.com/RaRe-Technologies/gensim) library.
The F1 result we achieved was 0.591 - probably one of the lowest so far. The respective Jupyter notebook 
is vectorizer_dense.ipynb. 
- Count vectorization which ended up being used for the topic modeling effort with generated notebook 
vectorizer_countVec.ipynb

After comparing the results, we ended up using the first vectorization option for our models - tweet tokenizer
with TF/IDF from sklearn, since it was producing the best evaluation scores.

## Classification algorithms selection and comparison

For the class label classification, we attempted the following algorithms and algorithm combinations:
- Linear regression - we achieved the 0.710 average macro F1 score leveraging a Grid search technique. 
- Random Forest - this algorithm was disappointing for this multiclass scenario with average macro F1 score
of 0.207. As a comparison, the dummy classifiers were giving scores from 0.04 to 0.08. 
- Multinomial NB - the F1 score was 0.569, better than random forest, but worse as compared to the logistic
regression.
- Voting classifier, where the three algorithms above were voting on the label. The averaged macro F1 score
on this one came to 0.689, again, not as good as the logistic regression, but better than the others.
We alo tried the Gradient Boosting Classifier, but we have to interrupt it, because it was taking too
long to train to be practical.

There is a separate section on the more advanced neural network algorithms that we tried. However, the 
results from them were very compatible with the logistic regression one. Therefore, the added complexity
did not bring any substantial benefit, and we decided to proceed with the simpler one for the rest of the 
project effort.

The complete exploration can be found in the category_classification_models.ipynb in the output folder of 
the project.

## Final observations and conclusions
At the end, and based on the observed scores and results, we decided to use the tweet tokenizer with a
multiclass classification logistic regression algorithm.