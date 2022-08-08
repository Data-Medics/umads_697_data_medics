

import os
import pandas as pd

import locations as loc

data_path = os.path.join(loc.data, "all_combined", "all_train.tsv")
training_data_sample = pd.read_csv(data_path, nrows=500, sep="\t")
training_data_sample.to_csv(os.path.join(loc.blog_data, "train_data_sample.csv"), index=False)














