# Recommended Actions

## Tweet Classication and Filtering
As discussed in the project introduction, information is extremely hard to come by during a natural disaster.  Depending on the type and scope of the disaster phone lines and other traditional means of communication may be disabled, making information via the internet, and specifically Twitter, extremely valuable.  However, Twitter is a massive ecosystem and it can be very difficult to use the correct search terms when looking for information, and equally difficult to find useful information within the massive amount of results even when proper search terms are used.  

Our project would be invaluable to those in the midst of a natural disaster by aggregating tweets, classifying them, and making them easier to search - in essence we built a natural disaster search engine on top of Twitter.  The information and resources revealed via our search tool could range from sites where donations were being collected to locations where resources and food could be picked up by those affected.  Below is a small demonstration of how our tool works: first, tweets relevant to natural disasters - collected via our disaster specific search term topics and classified as a relevant tweet via our classification model - are parsed, cleaned and stored with additional data points like `Geo-Political Entity` extracted and saved.  Next, a second classification model predicts which one of ten classes the tweet belongs to, adding another layer of searchability.  

Finally the tweets are aggregated and sorted by re-tweet count resulting in collection of relevant, curated, searchable, natural disaster specific tweets.

```
Interactive Code
```

## Actionable Insights
In addition to curating and aggregating useful tweets our project also extracts useful information from the tweets - specifically those tweets that suggest or 
recommend some concrete action - and makes it available in a easy to consume format.  Again, in the information overload that is Twitter it can be hard to 
easily find resources to access (for those involved in the disaster), or how/where to make donations and help (for those not directly involved in the disaster).  

To accomplish this we developed a three step process.  First, we queried current Twitter for natural disaster related information by 
searching for the most relevant topics and words for each disaster type, as determined by our topic models.  Next, we applied our classification model and kept
 only the tweets that were labeled as `rescue_volunteering_or_donation_effort`, hypothesizing that these tweets would be the most important for those searching 
 for resources or with resources to donate.  At this step we also deduplicated re-tweets and kept a track of the count, using this re-tweet count as our sorting 
 mechanism (most re-tweeted to least re-tweeted).  Finally, we used Spacy to extract the important information from the tweet in order to concisely display what action 
 the tweet was recommending and where the user could go (via hyperlink) to accomplish the goal.

Below we demonstrate how we have extracted the key information from the tweet while also preserving the original tweet.  We believe that by making it easy to access resources 
or contribute to a recovery effort we can lower the barrier to entry and get more people the help they need while simealtaneously increasing the number of people donation, 
volunteering or aiding in some other fashion.

```
Interactive Code
```