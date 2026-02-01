#!/usr/bin/env python
# coding: utf-8

# # TM10007 Assignment template -- ECG data

# ## Data loading and cleaning
# 
# Below are functions to load the dataset of your choice. After that, it is all up to you to create and evaluate a classification method. Beware, there may be missing values in these datasets. Good luck!

# In[17]:


# Run this to use from colab environment
get_ipython().system('git clone https://github.com/jveenland/tm10007_ml.git')

import zipfile
import os
import pandas as pd

with zipfile.ZipFile('/content/tm10007_ml/ecg/ecg_data.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/tm10007_ml/ecg')

data = pd.read_csv('/content/tm10007_ml/ecg/ecg_data.csv', index_col=0)

print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')

