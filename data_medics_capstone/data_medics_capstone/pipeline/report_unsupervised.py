# + tags=[]
import pandas as pd
import numpy as np

# + tags=["parameters"]
upstream = []
lookback_days = None

# + tags=["injected-parameters"]
# This cell was injected automatically based on your stated upstream dependencies (cell above) and pipeline.yaml preferences. It is temporary and will be removed when you save this notebook
lookback_days = 10
product = {"nb": "/Users/mboussarov/_umsi/Capstone/umads_697_data_medics/pipeline/output/report.ipynb"}


# + tags=[]
df = pd.read_csv('../data/all_dev.tsv', sep='\t')
df.head()

# + tags=[]
# Hello
# -


