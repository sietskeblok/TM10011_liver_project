# This script imports and processes data containing quantitative medical image features extracted from T2-weighted MRI

# Import and pre-processing data 
import warnings
import numpy as np
from sklearn.model_selection import train_test_split 
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler
from worcliver.load_data import load_data
warnings.filterwarnings("ignore") 

# Import data
data = load_data()

# Convert data to pickle file
data.to_pickle("data.pkl")

# Read data
data = pd.read_pickle("data.pkl")

# Split label from features
X = data.drop(columns=['label']) 
y = data['label']  

# Split data in train and testsets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert labels to binary values
y_train = y_train.replace({'benign': 0, 'malignant': 1}).astype(int)
y_test = y_test.replace({'benign': 0, 'malignant': 1}).astype(int)

# Convert data to pickle files
X_train.to_pickle("X_train.pkl")
X_test.to_pickle("X_test.pkl")
y_train.to_pickle("y_train.pkl")
y_test.to_pickle("y_test.pkl")

print("Data succesfully imported")