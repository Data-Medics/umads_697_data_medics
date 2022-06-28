import pandas as pd
import numpy as np

# + tags=["parameters"]
upstream = []
lookback_days = None
# -

df = pd.read_csv('../data/all_dev.tsv', sep='\t')
df.head()


