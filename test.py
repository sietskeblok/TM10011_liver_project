
#data importeren
from assignment import X_train, y_train

#labels omschrijven naar 0 en 1
y_train = y_train.replace({'benign': 0, 'malignant': 1})

#imports
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import RFECV, SelectKBest, SelectFromModel
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from turtle import pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, LabelEncoder
from scipy.stats import mannwhitneyu
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
'''
# Functie voor de Mann-Whitney U-test voor feature selectie
def mannwhitneyu_test(X_train, y_train):
    p_values = []
    for col in range(X_train.shape[1]):  # Itereren door de kolommen (features)
        group1 = X_train[y_train == 'benign', col]  # Benigne klasse
        group2 = X_train[y_train == 'malignant', col]  # Maligne klasse
        _, p_value = mannwhitneyu(group1, group2)
        p_values.append(p_value)
    return np.array(p_values)

print(np.unique(y_train, return_counts=True))
'''

#Mann-WHithtney U-test functie voor feature selectie
def mannwhitneyu_test(X, y):
    p_values = []
    for i in range(X.shape[1]):
        stat, p_value = mannwhitneyu(X[:, i][y == 0], X[:, i][y == 1])  # Mann-Whitney U-test
        p_values.append(p_value)
    return -np.array(p_values)  # Om hoger te score als de p-waarde lager is


# Maak de classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier( random_state=42),
    'SVM': SVC(kernel='linear') ,
}

# Feature Selectie methoden:
feature_selectors = {
    'Mann-Whitney U': SelectKBest(score_func=mannwhitneyu_test),  
    'RFECV': RFECV(estimator=RandomForestClassifier(random_state=42), step=20, 
                   cv=StratifiedKFold(3), scoring='accuracy')  
}

# Nested Cross-Validation instellingen
outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Buitenste loop
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Binnenste loop

# Lijst om de resultaten van de cross-validatie op te slaan
results = []

# Loop over classifiers en feature selectors
for clf_name, clf in classifiers.items():
    for selector_name, selector in feature_selectors.items():
        
        # Maak een pipeline met de feature selector en classifier
        pipeline = Pipeline([
            ('scaler', RobustScaler()),  # Schaal de data --> moet er nog uit
            ('feature_selection', selector),  
            ('classifier', clf)  
        ])
        
        # Binnenste cross-validatie voor feature selectie
        inner_scores = cross_val_score(pipeline, X_train, y_train_encoded, cv=inner_cv, scoring='accuracy')
        print(f"Inner CV accuracy for {clf_name} with {selector_name}: {inner_scores.mean():.4f} ± {inner_scores.std():.4f}")
        
        # Buitenste cross-validation voor model evaluatie
        outer_scores = cross_val_score(pipeline, X_train, y_train_encoded, cv=outer_cv, scoring='accuracy')
        print(f"Outer CV accuracy for {clf_name} with {selector_name}: {outer_scores.mean():.4f} ± {outer_scores.std():.4f}")
        
        # Resultaten opslaan
        results.append((clf_name, selector_name, outer_scores.mean(), outer_scores.std()))

# Print de eindresultaten
print("\nFinal Results (Outer CV accuracy):")
for result in results:
    print(f"{result[0]} with {result[1]}: Mean = {result[2]:.4f}, Std = {result[3]:.4f}")