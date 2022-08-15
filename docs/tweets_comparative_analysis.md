# Comparative analysis of tweet samples from the past week for different natural disasters

## Summary
The objective of training tweet classification models for the purpose of this project is to use them later
on a recent tweets, do an automated analysis and be able to answer questions like that:
- Is there an active disaster going on?
- What kind of disaster is going on - like wildfire, earthquake, flood, hurricane?
- What are the affected locations, are different locations affected together?
- For the active disasters, what categories are the tweets fit in - distress tweets, ask for donations etc.?
- What are the recommended actions in regard to the disaster?
- What is the disaster intensity over time, what are the peak days?
- Can we preview the tweets for a particular time and learn better what is going on?

The following elaboration explains how we can answer most of these questions.

For the purpose of this effort we have obtained and analyzed tweets for the week between ...

## Obtaining sample tweets by querying the Twitter APIs
The first step in analyzing the most resent disaster tweets is obtaining them from Twitter. The solution we
settled on to achieve that is as follows:
- Using the keyword tokens obtained from the unsupervised topic exploration of the disaster sample, we
query the Twitter API for the past six days (this is max time allowed to go back) and for the tweets up to
certain count (default set to 2000) for every two-hour period. The queries we are using for the respective 
disasters are:

| Disaster |Twitter Query  |
| :------- | :-------------- |
| Wildfire | wildfire (donate OR (death toll) OR burn OR fire OR forest OR damage) |
| Earthquake | earthquake (pray OR victim OR donate OR rescue OR damage OR magnitude) |
| Flood | flood (relief OR water OR donate OR (death toll) OR rescue OR destroy) |
| Hurricane | hurricane (donate OR storm OR cyclone OR damage OR evacuation OR destroy) |

And the number of tweets retrieved is about 144,000 per disaster, or about 576,000 total. The details of
the retrieval process are in the recent_tweets_*.ipynb notebooks in the project output folder.

## Assigning a class label on the retrieved disaster tweets
The obtained tweets are labeled with one of following categories using the vectorizer and the classification
algorithm prepared in the upstream steps:
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
The completed dataset is ready for further analysis

## Determining the dominating disaster in a category
The first question we are trying to answer on the newly obtained dataset is find out if there is an
active disaster, and what kind it is. For that purpose, we group all the tweets per disaster type and 
class label, and find the count of the tweets for 8-hour intervals. A sample of the result 
dataframe looks like that:

![Grouped Tweets](images/grouped_tweets.png)

When we create a boxplot for the tweets count distribution just by disaster type, we can conclude
that there is no visible difference in the mean and variance of the samples:

![Grouped Tweets](images/class_distribution.png)

We can statistically prove the same by running an f-test on the disaster distributions. The 
result is:
```buildoutcfg
F_onewayResult(statistic=0.3869087889615666, pvalue=0.7624738994702899)
```
The f-statistic is low and below 10, the p-value is high for the 10% significance, therefore,
we cannot reject the null hypothesis that the means of the individual disaster samples are the same.

How about if we filter the tweets for the 'displaced_people_and_evacuations' class label. In
that case the boxplot chart looks like that:

![Grouped Tweets](images/displaced_class_distribution.png)

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
(None, None)
```
Which points to the fact it is unable to find a disaster where the number tweets in this
category clearly exceed the other. 

## Final observations and conclusions