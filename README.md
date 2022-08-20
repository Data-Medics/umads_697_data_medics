# Disaster Tweets Pipeline Project

## Overview
The Disaster Tweets Pipeline Project does apply statistical analysis and machine learning techniques
using a labeled set of disaster tweets, and leverages the models to process a new set or recent tweets and 
answer questions like:
- Is there a disaster going on?
- What type of disaster it is?
- What are the affected locations?
- What are the recommended actions per the organizations that handle the disasters

In addition, the project provides an interactive visualization of the disaster intensity over time, and allows
a user to obtain additional insights.

The project also does an extensive elaboration on the supervised and unsupervised(topic modeling) process
used to train, tune and compare the models.

## Documentation
An elaboration on the exploratory details and results of the project can be found in our 
[Streamlit app](https://data-medics-umads-697-data-medics-blog-postblog-a49khp.streamlitapp.com/) and in
the [docs](https://github.com/Data-Medics/umads_697_data_medics/blob/main/docs/README.md) folder of this
repository. [Streamlit app](https://data-medics-umads-697-data-medics-blog-postblog-a49khp.streamlitapp.com/)
supports some interactive features and is the recommended place to start.

## Project organization
The exploration is leveraging a tool called [Ploomber](https://ploomber.readthedocs.io/en/latest/) for building ML pipelines.
The project code and results are in the folder [pipeline](https://github.com/Data-Medics/umads_697_data_medics/tree/main/pipeline) -
 it contains the configuration ([pipeline.yaml](https://github.com/Data-Medics/umads_697_data_medics/blob/main/pipeline/pipeline.yaml)) and 
the task logic in the respective *.py files. The "data" folder contains the files used for the 
exploration, and the "output" folder - the results of the pipeline execution as Jupyter notebooks and *.csv files once
the processing pipeline gets executed. 

Running all the tasks may take a long time - like a couple of hours. The command for executing that is:
```buildoutcfg
cd pipeline
ploomber build
```
You can also run just a specific pipeline task, it will execute the selected task and also all the dependent steps
that have not been executed yet. I.e. the following can take an hour to complete on a Mac with 16Gb of RAM
and 8-core CPU:
```buildoutcfg
ploomber task tweets_timeline_fire
```
This will recreate the files _tweets_timeline_fire.ipynb_ in the output folder.

## Project Setup

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

You also have to run the following command 

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
You may want to inspect and run the generated Jupyter notebooks or use the interactive charts. In that case,
you have to run Jupyter and navigate to the output folder to preview the outcome. Use the following
command from the "pipeline" project folder:

```buildoutcfg
jupyter notebook .
```
