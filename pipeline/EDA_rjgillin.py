# +
import pandas as pd
import numpy as np
import re


import matplotlib.pyplot as plt
from collections import Counter, defaultdict

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split 
from tqdm import tqdm # Used to show a progress bar

#tags= ["parameters"]
upstream = []


# -

random_seed = 10

# ## Load the data

df_dev = pd.read_csv('../data/HumAID_data_v1.0/all_combined/all_dev.tsv', sep='\t')
df_train = pd.read_csv('../data/HumAID_data_v1.0/all_combined/all_train.tsv', sep='\t')
df_test = pd.read_csv('../data/HumAID_data_v1.0/all_combined/all_test.tsv', sep='\t')

df_dev.head()

df_all = pd.concat([df_dev, df_train, df_test])

df_all.isnull().sum()

df_all.dropna(inplace=True)

len(df_all)

df_all['class_label'].unique()


