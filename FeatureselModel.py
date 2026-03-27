# Importeren modules
import warnings
warnings.filterwarnings("ignore") 

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import LinearSVC
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
from sklearn.model_selection import learning_curve
from sklearn.base import clone




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

X_train = pd.read_pickle("X_train.pkl")
X_test = pd.read_pickle("X_test.pkl")
y_train = pd.read_pickle("y_train.pkl")
y_test = pd.read_pickle("y_test.pkl")

#f2_scorer = make_scorer(fbeta_score, beta=2)

corr_filter = CorrelationFilter(threshold=0.95)
corr_filter.fit(X_train)
X_filtered = corr_filter.transform(X_train)

print(X_train.shape)
print(X_filtered.shape)
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
    'Logistic Regression': LogisticRegression(max_iter=7500, solver='saga', random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    #'rbfSVM': SVC(kernel='rbf', probability=True),
    'linearSVM': SVC(kernel='linear', probability=True)  
}

# Feature selectie modellen
feature_selectors = {
    'None': None,
    'Mann-Whitney U': SelectKBest(score_func=mannwhitneyu_test),
    'RFECV': RFECV(
        estimator=LogisticRegression(random_state=42, max_iter=5000),
        step=5, #steps moeten omlaag naar 1 eigenlijk
        cv=4,
        scoring='roc_auc'
    )
}

# Cross-validation
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) #folds moeten omhoog
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
            param_grid['classifier__penalty'] = ['l1','l2', 'elasticnet']
            param_grid['classifier__l1_ratio'] = [0.2, 0.5, 0.8]

        if clf_name == 'Random Forest':
            param_grid['classifier__n_estimators'] = [50, 100, 200, 300]
            param_grid['classifier__max_depth'] = [None, 5, 10]  
            param_grid['classifier__max_features'] = ['sqrt', 'log2', 0.2, 0.5] 

        elif clf_name == 'linearSVM':
            param_grid['classifier__C'] =  [0.01, 0.1, 1, 10]

        '''elif clf_name == 'rbfSVM':
            param_grid['classifier__C'] =  [0.01, 0.1, 1, 10]
            param_grid['classifier__gamma'] = ['scale', 'auto']'''

        if selector_name == 'Mann-Whitney U':
            param_grid['feature_selection__k'] = [5, 10, 15, 20]
        
        # GridSearch
        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=inner_cv,
            scoring='roc_auc',
            n_jobs=-1
        )

        # Outer CV
        outer_scores = cross_val_score(
            grid,
            X_train,
            y_train,
            cv=outer_cv,
            scoring='roc_auc',
            n_jobs=-1
        )

        all_outer_scores[(clf_name, selector_name)] = outer_scores
        mean_score = outer_scores.mean()
        std_score = outer_scores.std()

        print(f"Outer CV ROC_AUC-score: {mean_score:.4f} ± {std_score:.4f}")

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
#paar dingen checken
print("Classes:", best_grid.classes_)
print("Unique y_test:", np.unique(y_test, return_counts=True))

y_pred = best_grid.predict(X_test)
print("Pred distribution:", np.unique(y_pred, return_counts=True))

# Learning curve
train_sizes, train_scores, val_scores = learning_curve(
    estimator=best_grid,
    X=X_train,
    y=y_train,
    cv=outer_cv,
    scoring='roc_auc',
    train_sizes=np.linspace(0.1, 1.0, 5),
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, marker='o', label='Training score')
plt.plot(train_sizes, val_mean, marker='o', label='Validation score')

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)

plt.xlabel("Number of training samples")
plt.ylabel("ROC_AUC-score")
plt.title(f"Learning Curve: {best_name[0]} + {best_name[1]}")
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
#grid opslaan 
joblib.dump(best_grid, "best_model.pkl")
# %%
# Op testset testen
y_pred = best_grid.predict(X_test)
#y_proba = best_grid.predict_proba(X_test)[:, 1]
#y_pred = (y_proba > 0.6).astype(int)

print("\nBest model:")
print(f"{best_name[0]} with {best_name[1]}")

print("\nBest hyperparameters:")
for param, value in best_grid.best_params_.items():
    print(f"{param}: {value}")
    
 
# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# ROC-AUC (belangrijk!)
if hasattr(best_grid, "decision_function"):
    y_score = best_grid.decision_function(X_test)
else:
    y_score = best_grid.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_score)

# Confusion matrix → haalt TN, FP, FN, TP eruit
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Sensitiviteit en specificiteit berekenen
sensitivity = tp / (tp + fn)  # Sensitiviteit
specificity = tn / (tn + fp)  # Specificiteit

print("\nTest set performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"False Positives (benign → malignant): {fp}")
print(f"False Negatives (malignant → benign): {fn}")
print(f"ROC_AUC-score: {roc_auc:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")


#ROC curve code
y_proba = best_grid.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC curve\nNested CV ROC_AUC-score = {best_score:.2f} ± {best_std:.2f}")
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
plt.ylabel("ROC_AUC-score")
plt.title("Cross-validation ROC_AUC-score for all model combinations")
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


#ROC curve voor alle outer folds apart + mean ROC + std daaromheen

mean_fpr = np.linspace(0, 1, 100)
tprs = []
aucs = []

plt.figure(figsize=(8, 6))

for i, (train_idx, val_idx) in enumerate(outer_cv.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # opnieuw nested CV binnen deze outer fold
    model = clone(best_grid)
    model.fit(X_tr, y_tr)

    y_score = model.predict_proba(X_val)[:, 1]

    fpr, tpr, _ = roc_curve(y_val, y_score)
    fold_auc = roc_auc_score(y_val, y_score)

    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0

    tprs.append(interp_tpr)
    aucs.append(fold_auc)

    plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f"ROC fold {i} (AUC = {fold_auc:.2f})")

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
std_tpr = np.std(tprs, axis=0)

mean_auc = np.mean(aucs)
std_auc = np.std(aucs)

tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
tpr_lower = np.maximum(mean_tpr - std_tpr, 0)

plt.plot([0, 1], [0, 1], linestyle='--', color='red', lw=2, label='Chance')
plt.plot(mean_fpr, mean_tpr, color='blue', lw=2.5,
         label=f"Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})")
plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='grey', alpha=0.2,
                 label='± 1 std. dev.')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"Outer CV ROC curve: {best_name[0]} + {best_name[1]}")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close('all')