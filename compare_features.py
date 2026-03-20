#from MannwhitneyU import best_features_mwu
#from RFECV import selected_50_features

'''import numpy as np
from assignment import X_train, y_train

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, RFECV, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Labels encoden
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Pipeline met placeholders
pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('feature_selection', 'passthrough'),
    ('classifier', LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000))
])

# Inner en outer CV
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Mogelijke feature selectors
feature_selectors = [
    SelectKBest(score_func=f_classif, k=15),
    SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)),
    RFECV(
        estimator=RandomForestClassifier(n_estimators=100, random_state=42),
        step=20,
        cv=3,
        scoring='accuracy'
    )
]

# Mogelijke classifiers
classifiers = [
    LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000),
    RandomForestClassifier(n_estimators=100, random_state=42),
    SVC(kernel='linear')
]

# Parameter grid
param_grid = {
    'feature_selection': feature_selectors,
    'classifier': classifiers
}

# GridSearchCV = inner loop
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=inner_cv,
    scoring='accuracy',
    n_jobs=-1
)

# Outer loop
outer_scores = cross_val_score(
    grid_search,
    X_train,
    y_train_encoded,
    cv=outer_cv,
    scoring='accuracy',
    n_jobs=-1
)

print("Nested CV accuracy:")
print(f"Mean = {outer_scores.mean():.4f}")
print(f"Std  = {outer_scores.std():.4f}")
'''
# data importeren
from assignment import X_train, y_train

# imports
import numpy as np
from scipy.stats import mannwhitneyu

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.svm import SVC

# ----------------------------
# Mann-Whitney U-test functie
# ----------------------------
def mannwhitneyu_test(X, y):
    scores = []
    p_values = []

    class_0 = np.unique(y)[0]
    class_1 = np.unique(y)[1]

    for col in range(X.shape[1]):
        group1 = X[y == class_0, col]
        group2 = X[y == class_1, col]

        _, p_value = mannwhitneyu(group1, group2, alternative='two-sided')

        # SelectKBest wil: hoge score = beter
        scores.append(-p_value)
        p_values.append(p_value)

    return np.array(scores), np.array(p_values)

# ----------------------------
# Labels encoden
# ----------------------------
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

print("Klassenverdeling:")
print(np.unique(y_train, return_counts=True))

# ----------------------------
# Pipeline
# ----------------------------
pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('feature_selection', 'passthrough'),
    ('classifier', LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000))
])

# ----------------------------
# CV-instellingen
# ----------------------------
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# ----------------------------
# Parameter grids
# ----------------------------

# 1) MWU + classifiers
param_grid_mwu = [
    {
        'feature_selection': [SelectKBest(score_func=mannwhitneyu_test)],
        'feature_selection__k': [10, 15, 20],
        'classifier': [LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000)],
        'classifier__C': [0.1, 1, 10]
    },
    {
        'feature_selection': [SelectKBest(score_func=mannwhitneyu_test)],
        'feature_selection__k': [10, 15, 20],
        'classifier': [SVC(kernel='linear')],
        'classifier__C': [0.1, 1, 10]
    },
    {
        'feature_selection': [SelectKBest(score_func=mannwhitneyu_test)],
        'feature_selection__k': [10, 15, 20],
        'classifier': [RandomForestClassifier(random_state=42)],
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 5, 10]
    }
]

# 2) Greedy backward + classifiers
param_grid_backward = [
    {
        'feature_selection': [
            SequentialFeatureSelector(
                LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000),
                direction='backward',
                scoring='accuracy',
                cv=3,
                n_jobs=1
            )
        ],
        'feature_selection__n_features_to_select': [10, 15, 20],
        'classifier': [LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000)],
        'classifier__C': [0.1, 1, 10]
    },
    {
        'feature_selection': [
            SequentialFeatureSelector(
                LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000),
                direction='backward',
                scoring='accuracy',
                cv=3,
                n_jobs=1
            )
        ],
        'feature_selection__n_features_to_select': [10, 15, 20],
        'classifier': [SVC(kernel='linear')],
        'classifier__C': [0.1, 1, 10]
    },
    {
        'feature_selection': [
            SequentialFeatureSelector(
                LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000),
                direction='backward',
                scoring='accuracy',
                cv=3,
                n_jobs=1
            )
        ],
        'feature_selection__n_features_to_select': [10, 15, 20],
        'classifier': [RandomForestClassifier(random_state=42)],
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 5, 10]
    }
]

param_grid = param_grid_mwu + param_grid_backward

# ----------------------------
# GridSearchCV = inner loop
# ----------------------------
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=inner_cv,
    scoring='accuracy',
    n_jobs=1
)

# ----------------------------
# Outer loop = nested CV
# ----------------------------
outer_scores = cross_val_score(
    grid_search,
    X_train,
    y_train_encoded,
    cv=outer_cv,
    scoring='accuracy',
    n_jobs=1
)

print("\nNested CV accuracy:")
print(f"Mean = {outer_scores.mean():.4f}")
print(f"Std  = {outer_scores.std():.4f}")

# ----------------------------
# Beste combinatie op volledige trainingsdata
# ----------------------------
grid_search.fit(X_train, y_train_encoded)

best_selector = grid_search.best_params_['feature_selection']
best_classifier = grid_search.best_params_['classifier']

print("\nBeste combinatie op volledige trainingsset:")
print("Feature selector:", type(best_selector).__name__)
print("Classifier:", type(best_classifier).__name__)
print("Beste parameters:")
for key, value in grid_search.best_params_.items():
    print(f"  {key}: {value}")

print(f"\nBeste inner CV score: {grid_search.best_score_:.4f}")
from MannwhitneyU import significant_features
from RFECV import selected_50_features

set_1 = set(significant_features.index.tolist())
set_2 = set(selected_50_features)

# Vind de gemeenschappelijke features tussen de twee sets
common_features = set_1.intersection(set_2)

# Print het aantal gemeenschappelijke features
print(f"Aantal gemeenschappelijke features: {len(common_features)}")

# Print de lijst van gemeenschappelijke features
print(f"Gemeenschappelijke features: {common_features}")
