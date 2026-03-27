# This script contains the code for the TM10011 Liver Project (Group 7) and trains/tests the optimal ML model

# Import modules
import warnings
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
warnings.filterwarnings("ignore") 

# Load pre-processed training and test data
X_train = pd.read_pickle("X_train.pkl")
X_test = pd.read_pickle("X_test.pkl")
y_train = pd.read_pickle("y_train.pkl")
y_test = pd.read_pickle("y_test.pkl")

# Correlation filter to remove highly correlated features
class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold = 0.95):

        # Set correlation treshold above which a feature will be removed
        self.threshold = threshold
        self.to_drop_ = None
        self.columns_ = None

    def fit(self, X, y = None):

        # Convert input to dataframe 
        X = pd.DataFrame(X)
        self.columns_ = X.columns

        # Compute absolute correlation matrix
        corr_matrix = X.corr().abs()

        # Keep upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(bool))

        # Identify features to remove that have a correlation above treshold
        self.to_drop_ = [col for col in upper.columns if any(upper[col] > self.threshold)]

        return self

    def transform(self, X):

        # Convert input to dataframe
        X = pd.DataFrame(X)
        X.columns = self.columns_

        # Remove highly correlated columns
        return X.drop(columns = self.to_drop_, errors = 'ignore')

# Feature selection using Mann-Whitney U 
def mannwhitneyu_test(X, y):

    # Convert input data to numpy arrays
    X = np.asarray(X)
    y = np.asarray(y)

    p_values = [] # List to store p-values for each feature

    # Loop over each feature in X
    for i in range(X.shape[1]):

        # Perform Mann-Whitney U test between the two classes for feature i
        _, p_value = mannwhitneyu(X[y == 0, i], X[y == 1, i])

        # Append p-value to the list
        p_values.append(p_value)

    # Return negative p-values so that lower p-values (more significant features) get higher scores
    return -np.array(p_values) 

# Different types of classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter = 1000, solver = 'liblinear'),
    'Random Forest': RandomForestClassifier(random_state = 42),
    'linearSVM': SVC(kernel = 'linear', probability = True) 
}

# Different types of feature selection techniques
feature_selectors = {
    'None': None,
    'Mann-Whitney U': SelectKBest(score_func = mannwhitneyu_test),
    'RFECV': RFECV(
        estimator = LinearSVC(random_state=42),
        step = 10, # Remove 10 features per iteration
        cv = 4,
        scoring = 'roc_auc'
    )
}

# Cross-validation
outer_cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42) 
inner_cv = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 42) 

results = []

best_score = -np.inf
best_std = None
best_grid = None
best_name = None

# Store all results for visualization of boxplot
all_outer_scores = {}

# Nested cross-validation loop
for clf_name, clf in classifiers.items():
    for selector_name, selector in feature_selectors.items():

        # Print classifier and feature selection technique
        print(f"\nEvaluating {clf_name} with {selector_name}")

        # Machine learning pipeline 
        steps = [
            ('variance', VarianceThreshold(threshold=0.0)), # Remove features with no variability
            ('correlation', CorrelationFilter(threshold=0.95)), # Remove highly correlated features
            ('scaler', RobustScaler()), # Scale features robustly 
            ('feature_selection', selector), # Type of feature selection technique
            ('classifier', clf) # Type of classifier 
        ]

        pipeline = Pipeline(steps)

        # Define classifier hyperparameters
        param_grid = {}

        if clf_name == 'Logistic Regression':
            param_grid['classifier__C'] = [0.01, 0.1, 1, 10]
            param_grid['classifier__penalty'] = ['l2']

        elif clf_name == 'Random Forest':
            param_grid['classifier__n_estimators'] = [50, 100, 200, 300]
            param_grid['classifier__max_depth'] = [None, 5, 10]  
            param_grid['classifier__max_features'] = ['sqrt', 'log2', 0.2, 0.5] 

        elif clf_name == 'linearSVM':
            param_grid['classifier__C'] =  [0.01, 0.1, 1, 10]

        # Define feature selection hyperparameter
        if selector_name == 'Mann-Whitney U':
            param_grid['feature_selection__k'] = [5, 10, 15, 20]
        
        # Inner loop containing GridSearch
        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=inner_cv,
            scoring='roc_auc',
            n_jobs=-1
        )

        # Outer loop 
        outer_scores = cross_val_score(
            grid,
            X_train,
            y_train,
            cv=outer_cv,
            scoring='roc_auc',
            n_jobs=-1
        )

        # Store results
        all_outer_scores[(clf_name, selector_name)] = outer_scores
        mean_score = outer_scores.mean()
        std_score = outer_scores.std()

        print(f"Outer CV ROC_AUC-score: {mean_score:.4f} ± {std_score:.4f}")

        results.append((clf_name, selector_name, mean_score, std_score))

        # Store optimal parameters for machine learning model
        if mean_score > best_score:
            best_score = mean_score
            best_std = std_score
            best_grid = grid
            best_name = (clf_name, selector_name)

# Final machine learning model training 
print("\nFinal Results (Nested CV):")
for result in results:
    print(f"{result[0]} with {result[1]}: Mean = {result[2]:.4f}, Std = {result[3]:.4f}")

# Train optimal machine learning model on outer loop training data
best_grid.fit(X_train, y_train)

# Predict class labels for the test set
y_pred = best_grid.predict(X_test)

# Outer loop test set evaluation

# ROC-AUC 
if hasattr(best_grid, "decision_function"):
    y_score = best_grid.decision_function(X_test)
else:
    y_score = best_grid.predict_proba(X_test)[:, 1]

# Performance metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_score)

# Confusion matrix 
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Calculate sensitivity (recall) en specificity
sensitivity = tp / (tp + fn)  
specificity = tn / (tn + fp)  

print("\nTest set performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"False Positives (benign → malignant): {fp}")
print(f"False Negatives (malignant → benign): {fn}")
print(f"ROC_AUC-score: {roc_auc:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")

# Model performance visualization

# Learning curve visualization
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

# Save trained model by storing full pipeline 
joblib.dump(best_grid, "best_model.pkl")

# Predict class labels for the test set
y_pred = best_grid.predict(X_test)

# Print optimal machine learning model
print(f"\nBest model: {best_name[0]} with {best_name[1]}")

print("\nBest hyperparameters:")
for param, value in best_grid.best_params_.items():
    print(f"{param}: {value}")
    
# ROC AUC visualization

# Predict probabilities for the positive class (malignant = 1)
y_proba = best_grid.predict_proba(X_test)[:, 1]

# Compute ROC curve values (fpr = false positive rate, tpr = true positive rate)
fpr, tpr, _ = roc_curve(y_test, y_proba)

# Compute area under the curve
roc_auc = roc_auc_score(y_test, y_proba)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC curve\nNested CV ROC_AUC-score = {best_score:.2f} ± {best_std:.2f}")
plt.legend()
plt.show()

#ROC curve for all outer folds, mean ROC and standard deviations

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

# Boxplot visualization
labels = []
score_data = []

# Loop over all classifiers and feature selection techniques
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

# Confusion matrix visualization
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