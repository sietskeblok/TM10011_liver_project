#%% Forward Feature Selection using assignment.py objects

# Importeer de benodigde variabelen uit assignment.py
from assignment import X_train, X_test, y_train, scaler
import pandas as pd
import numpy as np

# Scaled training set
X_scaled = scaler.transform(X_train)  # Gebruik dezelfde scaler als in assignment.py

#%% 1️⃣ Variance filtering (alleen op training set)
from sklearn.feature_selection import VarianceThreshold

var_thresh = VarianceThreshold(threshold=0.01)
X_train_var = var_thresh.fit_transform(X_scaled)
features_var = X_train.columns[var_thresh.get_support()]

print(f"Number of features after variance filtering: {X_train_var.shape[1]}")

#%% 2️⃣ Correlation filtering (alleen op training set)
X_train_var_df = pd.DataFrame(X_train_var, columns=features_var)
corr_matrix = X_train_var_df.corr().abs()

# Bovenste driehoek van correlatiematrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]

X_train_filtered = X_train_var_df.drop(columns=to_drop)
filtered_features = X_train_filtered.columns
print(f"Number of features after correlation filtering: {X_train_filtered.shape[1]}")

#%% 3️⃣ Forward Selection met Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import StratifiedKFold

model = LogisticRegression(max_iter=1000)
sfs = SequentialFeatureSelector(
    estimator=model,
    n_features_to_select=10,  # maximaal aantal features
    direction="forward",
    scoring="roc_auc",
    cv=StratifiedKFold(3),
    n_jobs=-1
)
sfs.fit(X_train_filtered, y_train)

selected_features_forward = list(filtered_features[sfs.get_support()])
print("\nSelected features via Forward Selection:")
print(selected_features_forward)

#%% 4️⃣ Maak training en test sets met geselecteerde features
X_train_selected = X_train_filtered[selected_features_forward]

# Test set: transformeer met dezelfde scaler en behoud alleen geselecteerde features
X_test_scaled_df = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)
X_test_selected = X_test_scaled_df[selected_features_forward]

print("\nShape training set after forward selection:", X_train_selected.shape)
print("Shape test set after forward selection:", X_test_selected.shape)