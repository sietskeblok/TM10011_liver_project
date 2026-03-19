import numpy as np
from sklearn.model_selection import train_test_split 
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler
from worcliver.load_data import load_data

#data inladen
data = load_data()

print("data geladen")

# Verwijder label uit kolommen voor alleen features 
X = data.drop(columns=['label'])  # Verwijder zowel 'label' als 'ID' van de features
y = data['label']  # De targetvariabele is 'label'

# Splits de data in trainings- en testsets (bijv. 80% training en 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

initial_features = X.shape[1]

# 0 variantie verwijderen
var_selector = VarianceThreshold(threshold=0.0)

X_train_var = pd.DataFrame(
    var_selector.fit_transform(X_train),
    columns=X_train.columns[var_selector.get_support()],
    index=X_train.index
)

X_test_var = pd.DataFrame(
    var_selector.transform(X_test),
    columns=X_train_var.columns,
    index=X_test.index
)

removed_variance = initial_features - X_train_var.shape[1]

# Hoge correlaties verwijderen
corr_matrix = X_train_var.corr().abs()

upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]

X_train_filtered = X_train_var.drop(columns=to_drop)
X_test_filtered = X_test_var.drop(columns=to_drop)

removed_corr = len(to_drop)
'''
# Print de resultaten van de pre-processing
print(f"Initial features: {initial_features}")
print(f"Removed zero variance: {removed_variance}")
print(f"Removed high correlation: {removed_corr}")
print(f"Final features: {X_train_filtered.shape[1]}")
'''

# Data scalen
scaler = RobustScaler()
X_train_filtered_scaled = scaler.fit_transform(X_train_filtered)
X_test_filtered_scaled = scaler.transform(X_test_filtered)

# Data converteren naar df
X_train_filtered_scaled = pd.DataFrame(
    X_train_filtered_scaled,
    columns=X_train_filtered.columns,
    index=X_train_filtered.index
)

X_test_filtered_scaled = pd.DataFrame(
    X_test_filtered_scaled,
    columns=X_train_filtered.columns,
    index=X_test_filtered.index
)

#omzetten naar binary
y_train = y_train.replace({'benign': 0, 'malignant': 1})
y_test = y_test.replace({'benign': 0, 'malignant': 1})

#Data opslaan als pickle 
X_train_filtered_scaled.to_pickle("X_train_filtered_scaled.pkl")
X_test_filtered_scaled.to_pickle("X_test_filtered_scaled.pkl")
y_train.to_pickle("y_train.pkl")
y_test.to_pickle("y_test.pkl")

print("data opgeslagen")

''' plak dit in de file waar je data in wil lezen
X_train = pd.read_pickle("X_train_filtered_scaled.pkl")
X_test = pd.read_pickle("X_test_filtered_scaled.pkl")
y_train = pd.read_pickle("y_train.pkl")
y_test = pd.read_pickle("y_test.pkl")
'''