#%% Main code Liver assignment 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split 
import seaborn as sns
import matplotlib.pyplot as plt

from worcliver.load_data import load_data
data = load_data()

# Verwijder label uit kolommen voor alleen features 
X = data.drop(columns=['label'])  # Verwijder zowel 'label' als 'ID' van de features
y = data['label']  # De targetvariabele is 'label'

# Splits de data in trainings- en testsets (bijv. 80% training en 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# scaling 
from sklearn.preprocessing import RobustScaler

# Maak een instantie van de RobustScaler
scaler = RobustScaler()

# Pas de scaler toe op de trainingsdata
X_scaled = scaler.fit_transform(X_train)

import seaborn as sns
import matplotlib.pyplot as plt
