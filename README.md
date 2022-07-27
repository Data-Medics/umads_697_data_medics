# umads_697_data_medics

## Project organization
The exploration is leveraging a tool called [Ploomber](https://ploomber.readthedocs.io/en/latest/) for building ML pipelines.
The project code and results are in the folder "pipeline" - it contains the pipeline configuration (pipeline.yaml) and 
the task logic in the respective *.py files. The "data" folder contains the files used for the 
exploration, and the "output" folder - the results of the pipeline execution as Jupyter notebooks and *.csv files. 
If you run Jupyter locally, you can navigate to the "output" folder and inspect the outcome without running the pipeline.

Running all the tasks may take a long time - like a day or more. The command for executing that is:
```buildoutcfg
cd pipeline
ploomber build
```
You can also run just a specific task, i.e. the following produces the best result with about 73%
accuracy, and it is intentionally made independent of the other tasks. It takes about 3 hours to complete
on a Mac with 16Gb of RAM and 8-core CPU:
```buildoutcfg
ploomber task supervised_computed_features
```
This will recreate the files _supervised_computed_features.ipynb_ and _supervised_computed_features.csv_
in the output folder.

## Setup

### Get the project

```buildoutcfg
git clone https://github.com/Data-Medics/umads_697_data_medics.git
cd umads_697_data_medics
```

### Setup Poetry

We use [poetry](https://python-poetry.org/docs/) to manage python dependency.
If you do not have poetry, please use [this command](https://python-poetry.org/docs/#osx--linux--bashonwindows-install-instructions)

In terminal, go to your project directory and update the poetry env (this is because we started off with the `pyproject.toml`). A new file will be created, poetry.lock 
```bash
$ poetry update 
```

Enter into poetry shell. With the shell you can run ploomner, or a Jupyter notebook. 
```bash
$ poetry shell
```

## Getting the data
Use the following [Google Drive location](https://drive.google.com/file/d/1pNMVhe1eXrm85SS6uyJkq8XVhF4Uqz2i/view) to obtain the data file.
Once you do, copy it to a folder <umads_697_data_medics project toot>/data. Execute the following commands:
```bash
$ gunzip HumAID_data_v1.0.tar.gz
$ tar -xvzf HumAID_data_v1.0.tar
```
This will create the following file structure that is used by the ML processing pipelines:
```
data
    HumAID_data_v1.0
        all_combined
        event_type
        events
```

## Running Jupyter Notebooks
```buildoutcfg
jupyter notebook .
```

## Pipeline processing logic
The pipelines executed with 'ploomber build" from the 'pipeline' folder perform the following:
- Vectorizing the tweets 
  - TF/IDF sparse vectors with bigrams
    - Add stemming, better stop words removal
  - Word embedding dense vectors with Word2vec
- Predicting the class labels from the tweet payload
  - Implemented basic linear regression
    - Achieved F1 score of 0.71 using TF/IDF vectors with bigrams
    - Word embedding dense vectors produced F1 of 0.63
  - Implement other algorithms (Random Forest, XGBoost, MultinomialNB, Voting Classifier), see if they improve the score
  - Implement a Neural Network see if it improves the score
- Predicting the disaster types from the tweet payload
  - Implemented basic linear regression
    - Achieved F1 score of 0.95 using TF/IDF vectors with bigrams
- Predict if a tweets is a disaster/nom-disaster tweet, leverage the category labels for that
- Do a topic modeling based on the tweets (elaborate)
- Obtain a sample of the recent disaster-related tweets (past one week) using the topic keywords: 
    - TODO: Identifying what disasters are going in the world, what is the intensity, are there some dominant disasters
    - Visualizing the locations of the tweets on the world map, showing where the disasters are happening based on a tweet stream
    - Categorizing the tweets for a disaster - i.e. caution and advise, infrastructure damage, resque effort 
    - Visualizing the locations affected, what locations are connected, what category they belong to
    - Visualizing how the disasters for the top progress over time, when is the peaks, how the categories change
- TODO: Explaining why the tweet is predicted as a label, what words contributed for that
- TODO: Aggregate and explain what features are contributing for the class predictions 
- Extract the actions to be performed when a disaster is going on:
  - Extract the names of the organizations that deal with the disasters
  - TODO: Extract the actions these organizations recommend/advise (the verbs)
  - TODO: Extract the respective objects for these actions (i.e. destinations etc)

## The complete functionality to be delivered (TODO)
- Produce and train the following:
  - Vectorizer that converts the tweet body to matrix/vectors
  - Classifiers that predict the following:
    - If this is a disaster tweet or not
    - What class is this tweet
    - What disaster type is this tweet about
  - Location extractor that identifies the main locations mentioned in a body of tweets
  - Organizations and the main actions from them as found in a body of tweets
- Extract a body of tweets by querying Tweeter APIs
- Apply the above classifiers on this body answering the following:
  - Is there a disaster going on?
  - What type of disaster it is?
  - What are the affected locations?
  - What are the recommended actions per the organizations that handle the disasters
  - Visualize the disaster intensity over time
- If time permits, load the data in [Streamlit](https://streamlit.io/), make the above analysis as an online app running there


