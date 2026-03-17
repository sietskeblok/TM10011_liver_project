#%% Feature Selection with RFECV (Greedy Backward Elimination)
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV

from worcliver.load_data import load_data

#%% Load data
data = load_data()

# Split features and target
X = data.drop(columns=['label'])
y = data['label']

#%% Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#%% Scaling
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%% RFECV: Greedy Backward Feature Elimination
model = LogisticRegression(max_iter=1000)

rfecv = RFECV(
    estimator=model,
    step=5,  # verwijder 5 features per iteratie (sneller bij 493 features)
    cv=StratifiedKFold(5),
    scoring="roc_auc",
    min_features_to_select=5,
    n_jobs=-1
)

rfecv.fit(X_train_scaled, y_train)

#%% Original selected features from RFECV
all_selected_features = X_train.columns[rfecv.support_]
all_selected_ranking = rfecv.ranking_[rfecv.support_]

# Combine features and ranking
feature_ranking_list = list(zip(all_selected_features, all_selected_ranking))

# Sorteer op ranking (1 = beste)
feature_ranking_list_sorted = sorted(feature_ranking_list, key=lambda x: x[1])

# Beperk tot maximaal 15 features
max_features = 15
selected_features_list = [f[0] for f in feature_ranking_list_sorted[:max_features]]

# Print resultaten
print(f"\nOptimal number of features suggested by RFECV: {rfecv.n_features_}")
print(f"Selected features (max {max_features}):")
print(selected_features_list)

#%% Create datasets with only selected features
X_train_selected = X_train[selected_features_list]
X_test_selected = X_test[selected_features_list]

print("\nShape training set after feature selection:", X_train_selected.shape)
print("Shape test set after feature selection:", X_test_selected.shape)

#%% Plot cross-validation performance
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (ROC AUC)")
plt.plot(
    range(rfecv.min_features_to_select,
          rfecv.min_features_to_select + len(rfecv.cv_results_["mean_test_score"])),
    rfecv.cv_results_["mean_test_score"]
)
plt.title("RFECV Feature Selection Performance")
plt.show()