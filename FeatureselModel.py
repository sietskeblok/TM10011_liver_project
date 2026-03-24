# Importeren modules
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
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
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay


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
    'None': None,
    'Mann-Whitney U': SelectKBest(score_func=mannwhitneyu_test),
    'RFECV': RFECV(
        estimator=RandomForestClassifier(random_state=42),
        step=30, #steps moeten omlaag naar 1 eigenlijk
        cv=4,
        scoring=f2_scorer
    )
}

# Cross-validation
outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) #folds moeten omhoog
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) #folds moeten omhoog

results = []

best_score = -np.inf
best_std = None
best_grid = None
best_name = None
all_outer_scores = {}

# Nested CV loop maken
for clf_name, clf in classifiers.items():
    for selector_name, selector in feature_selectors.items():

        print(f"\nEvaluating {clf_name} with {selector_name}")

        #Pipeline opbouwen
        steps = [
            ('variance', VarianceThreshold(threshold=0.0)),
            ('correlation', CorrelationFilter(threshold=0.95)),
            ('scaler', RobustScaler()), 
            ('feature_selection', selector),
            ('classifier', clf) 
        ]

        pipeline = Pipeline(steps)

        # Hyperparameters
        param_grid = {}

        if clf_name == 'Logistic Regression':
            param_grid['classifier__C'] = [0.01, 0.1, 1, 10]

        elif clf_name == 'Random Forest':
            param_grid['classifier__n_estimators'] = [50, 100, 200]
            param_grid['classifier__max_depth'] = [None, 5, 10]

        elif clf_name == 'SVM':
            param_grid['classifier__C'] = [0.01, 0.1, 1, 10]

        if selector_name == 'Mann-Whitney U':
            param_grid['feature_selection__k'] = [5, 10, 15, 20]

        # GridSearch
        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=inner_cv,
            scoring=f2_scorer,
            n_jobs=-1
        )

        # Outer CV
        outer_scores = cross_val_score(
            grid,
            X_train,
            y_train,
            cv=outer_cv,
            scoring=f2_scorer,
            n_jobs=-1
        )

        all_outer_scores[(clf_name, selector_name)] = outer_scores
        mean_score = outer_scores.mean()
        std_score = outer_scores.std()

        print(f"Outer CV F2-score: {mean_score:.4f} ± {std_score:.4f}")

        results.append((clf_name, selector_name, mean_score, std_score))

        if mean_score > best_score:
            best_score = mean_score
            best_std = std_score
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
joblib.dump(best_grid, "best_model.pkl")
# %%
# Op testset testen
y_pred = best_grid.predict(X_test)

print("\nBest model:")
print(f"{best_name[0]} with {best_name[1]}")

print("\nBest hyperparameters:")
for param, value in best_grid.best_params_.items():
    print(f"{param}: {value}")
    
# Accuracy and f2-score 
accuracy = accuracy_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2)

# Confusion matrix → haalt TN, FP, FN, TP eruit
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Sensitiviteit en specificiteit berekenen
sensitivity = tp / (tp + fn)  # Sensitiviteit
specificity = tn / (tn + fp)  # Specificiteit

print("\nTest set performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"False Positives (benign → malignant): {fp}")
print(f"False Negatives (malignant → benign): {fn}")
print(f"F2-score: {f2:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")


#ROC code
y_proba = best_grid.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC curve\nNested CV F2-score = {best_score:.2f} ± {best_std:.2f}")
plt.legend()
plt.show()
# %%
#boxplot
labels = []
score_data = []

for (clf_name, selector_name), scores in all_outer_scores.items():
    labels.append(f"{clf_name}\n+ {selector_name}")
    score_data.append(scores)

plt.figure(figsize=(10, 6))
plt.boxplot(score_data, tick_labels=labels)
plt.ylabel("F2-score")
plt.title("Cross-validation F2-score for all model combinations")
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.show()

#confusion matrix
ConfusionMatrixDisplay.from_estimator(
    best_grid,
    X_test,
    y_test,
    display_labels=["Benign", "Malignant"],
    cmap="Blues",
    normalize='true'
)

plt.title("Confusion Matrix (Normalized)")
plt.show()