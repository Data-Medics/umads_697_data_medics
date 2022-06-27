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
```buildoutcfg
git clone https://github.com/Data-Medics/umads_697_data_medics.git
cd umads_697_data_medics
python3 -m venv env
source env/bin/activate
./setup_local
cd pipeline
export PYTHONPATH=`pwd`:`pwd`/pipeline:$PYTHONPATH
```

## Running Jupyter Notebooks
```buildoutcfg
jupyter notebook .
```
