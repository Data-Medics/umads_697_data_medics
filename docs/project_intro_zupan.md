# Project Introduction


### Overview

Quality information is incredibly difficult to obtain following a major natural disaster and finding particular information can make a huge difference to those affected.  Lots of pertinent information is posted on Twitter following natural disasters, but it can be extremely challenging to sift through hundreds of thousands of tweets for the desired information.  To help solve this problem we built a one-stop-natural-disaster-information-shop.  Powered by our machine learning models, we make it easy to filter tweets into specific categories based on the information the user is trying to find.  We use natural language processing and topic models to search Twitter for tweets specific to different types of natural disasters.  From here we provide a wide variety of carefully curated, useful information ranging from geo-spatial analysis to understand the exact locations and severity of the natural disaster to parsing the Twitterverse for actionable behaviors.  The period following a natural disaster is incredibly difficult and dangerous for all those involved - our tool aims to make the recovery process easier and more efficient with better outcomes for those affected.
    
All models in this project were trained on a data set collected and curated by HumAid.  The data consists of ~70,000 tweets across 20 natural disasters spanning from 2016 to 2018.  The tweets were labeled by human assistants and are classified as belonging to one of the following 10 categories:
* Caution and advice
* Displaced people and evacuations
* Dont know cant judge
* Infrastructure and utility damage
* Injured or dead people
* Missing or found people
* Not humanitarian
* Other relevant information
* Requests or urgent needs
* Rescue volunteering or donation effort
* Sympathy and support

### Real Time Data

However, giving people critical information about a disaster that happened several years ago is of limited value.  For our project to truly help, we needed to apply our models and analysis to live Twitter data around current natural disasters.  To do this we utilized [Tweepy](https://www.tweepy.org/) and wrote several functions that pulled real time Twitter based on our disaste specific search terms which we then used as inputs to our model.
