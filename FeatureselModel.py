# Importeren modules
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.feature_selection import RFECV, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from scipy.stats import mannwhitneyu
from sklearn.svm import SVC
import pandas as pd

#data inladen
X_train = pd.read_pickle("X_train_filtered_scaled.pkl")
X_test = pd.read_pickle("X_test_filtered_scaled.pkl")
y_train = pd.read_pickle("y_train.pkl")
y_test = pd.read_pickle("y_test.pkl")

print("data ingeladen")
#print (X_train)
#print (y_train)



# Mann Whitney-U feature selectie
def mannwhitneyu_test(X, y):
    p_values = []
    for i in range(X.shape[1]):
        stat, p_value = mannwhitneyu(X[:, i][y == 0], X[:, i][y == 1])
        p_values.append(p_value)
    return -np.array(p_values)  # lagere p = hogere score

# Verschillende classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, solver='liblinear'),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(kernel='linear')
}

# Feature selectie
feature_selectors = {
    'Mann-Whitney U': SelectKBest(score_func=mannwhitneyu_test),
    'RFECV': RFECV(
        estimator=RandomForestClassifier(random_state=42),
        step=20,
        cv=4,
        scoring='accuracy'
    )
}

# Cross validation
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

# For loop voor feature selectie, classifier, hyperparameter tuning en validatie
for clf_name, clf in classifiers.items():
    for selector_name, selector in feature_selectors.items():

        print(f"\nEvaluating {clf_name} with {selector_name}")

        # pipeline
        pipeline = Pipeline([
            ('scaler', RobustScaler()), #dit moet er nog uit
            ('feature_selection', selector),
            ('classifier', clf)
        ])

        # Hyperparameters per classifier
        param_grid = {}

        # Logistic Regression
        if clf_name == 'Logistic Regression':
            param_grid['classifier__C'] = [0.01, 0.1, 1, 10]

        # Random Forest
        elif clf_name == 'Random Forest':
            param_grid['classifier__n_estimators'] = [50, 100, 200]
            param_grid['classifier__max_depth'] = [None, 5, 10]

        # SVM
        elif clf_name == 'SVM':
            param_grid['classifier__C'] = [0.01, 0.1, 1, 10]

        # Mann-Whitney: aantal features tunen
        if selector_name == 'Mann-Whitney U':
            param_grid['feature_selection__k'] = [5, 10, 15, 20]

        # Grid search (inner loop)
        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=inner_cv,
            scoring='accuracy',
            n_jobs=-1
        )

        # Outer loop
        outer_scores = cross_val_score(
            grid,
            X_train,
            y_train,
            cv=outer_cv,
            scoring='accuracy',
            n_jobs=-1
        )

        print(f"Outer CV accuracy: {outer_scores.mean():.4f} ± {outer_scores.std():.4f}")

        results.append((clf_name, selector_name, outer_scores.mean(), outer_scores.std()))

# Print resultaten
print("\nFinal Results (Nested CV):")
for result in results:
    print(f"{result[0]} with {result[1]}: Mean = {result[2]:.4f}, Std = {result[3]:.4f}")