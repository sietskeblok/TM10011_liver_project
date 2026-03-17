
from assignment import X_train, y_train

from sklearn.feature_selection import SelectKBest
from scipy.stats import mannwhitneyu
import pandas as pd

# Definieer de functie die de Mann-Whitney U-test uitvoert voor univariate feature selectie
def mannwhitneyu_test(X_train, y_train):
    # Resultaten van de Mann-Whitney U-test voor elke feature
    p_values = []
    for col in X_train.columns:
        # Mann-Whitney U-test tussen de twee klassen
        group1 = X_train[col][y_train == 'benign']  # Benigne klasse
        group2 = X_train[col][y_train == 'malignant']  # Maligne klasse
        _, p_value = mannwhitneyu(group1, group2)
        p_values.append(p_value)
    return pd.Series(p_values, index=X_train.columns)

# Voer de Mann-Whitney U-test uit om de p-waarden te krijgen
print("Running Mann-Whitney U test...")
p_values = mannwhitneyu_test(X_train, y_train)

# Selecteer de top 50 features met de laagste p-waarden (d.w.z. meest significante features)
best_features_mwu = p_values.nsmallest(50)
print(f"The 50 best features selected by Mann-Whitney U test are: {best_features_mwu.index.tolist()}")

# Selecteer de top 50 p-waarden (de laagste p-waarden, dus meest significante)
top_50_p_values = p_values.nsmallest(50)

# Print alleen de p-waarden van de 50 geselecteerde features
print("\nTop 50 p-values (lowest p-values):")
print(top_50_p_values.values)