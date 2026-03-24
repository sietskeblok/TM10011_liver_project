import numpy as np
from sklearn.model_selection import train_test_split 
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler

#data inladen
data = pd.read_pickle("data.pkl")

# Verwijder label uit kolommen voor alleen features 
X = data.drop(columns=['label'])  # Verwijder zowel 'label' als 'ID' van de features
y = data['label']  # De targetvariabele is 'label'

# Splits de data in trainings- en testsets (bijv. 80% training en 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

initial_features = X.shape[1]

y_train = y_train.replace({'benign': 0, 'malignant': 1}).astype(int)
y_test = y_test.replace({'benign': 0, 'malignant': 1}).astype(int)

X_train.to_pickle("X_train.pkl")
X_test.to_pickle("X_test.pkl")
y_train.to_pickle("y_train.pkl")
y_test.to_pickle("y_test.pkl")
