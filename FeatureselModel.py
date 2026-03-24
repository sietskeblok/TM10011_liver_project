#from assignment import X_train, y_train


#HIER STAAT PREPROCESSING NOG NIET IN DE NESTED CV
# Importeren modules
'''import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.feature_selection import RFECV, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from scipy.stats import mannwhitneyu
from sklearn.svm import SVC

X_train = pd.read_pickle("X_train_filtered_scaled.pkl")
X_test = pd.read_pickle("X_test_filtered_scaled.pkl")
y_train = pd.read_pickle("y_train.pkl")
y_test = pd.read_pickle("y_test.pkl")
#print(y_train.dtype)
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
        step=30,
        cv=4,
        scoring='accuracy'
    )
}

# Cross validation
outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

results = []

# For loop voor feature selectie, classifier, hyperparameter tuning en validatie
for clf_name, clf in classifiers.items():
    for selector_name, selector in feature_selectors.items():

        print(f"\nEvaluating {clf_name} with {selector_name}")

        # pipeline
        pipeline = Pipeline([
            #('scaler', RobustScaler()), #dit moet er nog uit
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
'''
# %%
#POGING 2, HIER STAAT PREPROCESSING WEL IN DE NESTED CV
# Importeren modules
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.feature_selection import RFECV, SelectKBest, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import make_scorer, fbeta_score

X_train = pd.read_pickle("X_train.pkl")
X_test = pd.read_pickle("X_test.pkl")
y_train = pd.read_pickle("y_train.pkl")
y_test = pd.read_pickle("y_test.pkl")

f2_scorer = make_scorer(fbeta_score, beta=2)

#correlatiefilter
class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.to_drop_ = None
        self.columns_ = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = X.columns

        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop_ = [col for col in upper.columns if any(upper[col] > self.threshold)]

        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        X.columns = self.columns_
        return X.drop(columns=self.to_drop_, errors='ignore')



# Mann-Whitney U feature selectie definitie
def mannwhitneyu_test(X, y):
    X = np.asarray(X)
    y = np.asarray(y)

    p_values = []
    for i in range(X.shape[1]):
        _, p_value = mannwhitneyu(X[y == 0, i], X[y == 1, i])
        p_values.append(p_value)

    return -np.array(p_values)  # lagere p = hogere score



# Verschillende classifiers in ons model
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, solver='liblinear'),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(kernel='linear', probability=True) #dit nog aanpassen naar polynoom?
}



# Feature selectie modellen
feature_selectors = {
    'Mann-Whitney U': SelectKBest(score_func=mannwhitneyu_test),
    'RFECV': RFECV(
        estimator=RandomForestClassifier(random_state=42),
        step=30, #deze moeten omlaag
        cv=4,
        scoring=f2_scorer
    )
}

# Cross-validation
outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) #folds moeten omhoog
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) #folds moeten omhoog

results = []

best_score = -np.inf
best_grid = None
best_name = None


# Nested CV loop maken
for clf_name, clf in classifiers.items():
    for selector_name, selector in feature_selectors.items():

        print(f"\nEvaluating {clf_name} with {selector_name}")

        # Pipeline met preprocessing BINNEN de CV-loop
        pipeline = Pipeline([
            ('variance', VarianceThreshold(threshold=0.0)),
            ('correlation', CorrelationFilter(threshold=0.95)),
            ('scaler', RobustScaler()),
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
            scoring=f2_scorer,
            n_jobs=-1
        )

        # Outer loop
        outer_scores = cross_val_score(
            grid,
            X_train,
            y_train,
            cv=outer_cv,
            scoring=f2_scorer,
            n_jobs=-1
        )

        mean_score = outer_scores.mean()
        std_score = outer_scores.std()

        print(f"Outer CV F2-score: {mean_score:.4f} ± {std_score:.4f}")

        results.append((clf_name, selector_name, mean_score, std_score))

        # Beste model bewaren
        if mean_score > best_score:
            best_score = mean_score
            best_grid = grid
            best_name = (clf_name, selector_name)


# Resultaten nested CV
print("\nFinal Results (Nested CV):")
for result in results:
    print(f"{result[0]} with {result[1]}: Mean = {result[2]:.4f}, Std = {result[3]:.4f}")


# Beste model opnieuw fitten op hele trainingsset, want je hebt nog niet getrained op de hele trainset
best_grid.fit(X_train, y_train)
# %%
#grid opslaan 
import joblib
joblib.dump(best_grid, "best_model.pkl")
# %%
best_grid = joblib.load("best_model.pkl")
# Op testset testen
y_pred = best_grid.predict(X_test)

print("\nBest model:")
print(f"{best_name[0]} with {best_name[1]}")

# Accuracy and f2-score 
accuracy = accuracy_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

# Confusion matrix → haalt TN, FP, FN, TP eruit
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("\nTest set performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"False Positives (benign → malignant): {fp}")
print(f"False Negatives (malignant → benign): {fn}")
print(f"F2-score: {f2:.4f}")


# %%
#ROC code
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

#print(best_grid)
y_proba = best_grid.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve")
plt.legend()
plt.show()