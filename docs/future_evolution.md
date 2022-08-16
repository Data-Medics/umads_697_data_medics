# Final summary and future evolution of the system

With the disaster tweets analysis project we did explore various ML models
capable of predicting a label of an individual tweets, determining that the simpler one
(Logistic Regression) did as good of a job for a multiclass classification. Furthermore,
we did an unsupervised topic modeling and extracted topic terms/tokens that we used to
query recent disaster tweets, then identified which disaster in a disaster category is 
dominant for a particular timeframe and what are the present active disasters. In addition,
we extracted the locations from a sample of recent tweets and provided various interactive
visualization and relationships between different locations, and also collected and showed
different actions as recommended by the tweet authors. In essence, we build a system
that can build and present a quite accurate picture of what is going on in the world in terms
of natural disasters based on the dynamics of the tweet info.

Some possible ways this system can evolve in the future: 
- __Building a batch or a near-realtime system__ that pulls the tweets from the past hour and
analyses them, publishing a report with the result. This will allow someone to follow up
daily of what is going on in the world.
- __Collecting and interpreting the disaster information over longer period of time__ - like months and
years. Unfortunately Twitter limits how far back in time you can go through the APIs and obtain the
relevant info. Therefore, some additional storage needs to be added, so the long-term data are available
for additional analysis and insights. 
- __Doing a realtime alerting__ when some truly disastrous/important happen with location details and other info
- __Adding a multilingual support__ - presently the system interprets tweets just in English,
and as a result it may be missing important information for locations and countries that 
use different languages.
- __Extracting other Twitter information, not just for disasters__, but for other scenarios -
geo-political, stock market, energy prices, government actions etc. 
- __Interpreting information not just from tweets, but other news sources__ - theoretically
other NLP sources are not that different as compared to the tweets. So they can be used as
a complimentary or a primary source of the required information.