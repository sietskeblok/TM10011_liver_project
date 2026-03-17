import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report, roc_curve

from assignment import X_train, X_test, X_scaled, y_train, y_test

y_train_num = y_train.map({'benign': 0, 'malignant': 1})
y_test_num = y_test.map({'benign': 0, 'malignant': 1})

param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'class_weight': [None, 'balanced']
}

model = LogisticRegression(
    solver='liblinear',      # liblinear ondersteunt L1 regularisatie
    penalty='l1',            # L1 regularisatie
    max_iter=10000,
    random_state=42
)

grid = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

# --- Fit op de geschaalde trainingsdata ---
grid.fit(X_scaled, y_train_num)

best_model = grid.best_estimator_
print("Beste parameters:", grid.best_params_)

# --- Feature selectie op basis van non-zero coëfficiënten ---
coefficients = best_model.coef_[0]
selected_features = np.array(X_train.columns)[coefficients != 0]

print("\nAantal originele features:", X_train.shape[1])
print("Aantal geselecteerde features:", len(selected_features))

print("\nGeselecteerde features:")
for feature in selected_features:
    print(feature)

# --- Trainings- en testdata beperken tot geselecteerde features ---
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# --- Final model trainen op geselecteerde features ---
final_model = LogisticRegression(
    solver='liblinear',
    penalty='l1',
    C=grid.best_params_['C'],
    class_weight=grid.best_params_['class_weight'],
    max_iter=5000,
    random_state=42
)

final_model.fit(X_train_selected, y_train_num)
print("\nModel getraind op geselecteerde features.")

# --- Predicties en predictie probabilities ---
y_pred = final_model.predict(X_test_selected)
y_pred_proba = final_model.predict_proba(X_test_selected)[:, 1]

# --- Evaluatie ---
roc_auc = roc_auc_score(y_test_num, y_pred_proba)
acc = accuracy_score(y_test_num, y_pred)
cm = confusion_matrix(y_test_num, y_pred)
report = classification_report(y_test_num, y_pred)

print("\nTest ROC-AUC:", roc_auc)
print("Test accuracy:", acc)
print("Confusion matrix:\n", cm)
print("Classification report:\n", report)
