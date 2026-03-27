# This script uses the Mann-Whitney U test to find features that differ significanlty between benign and malignant.

# Import modules
from ImportData import X_train, y_train
from sklearn.feature_selection import SelectKBest
from scipy.stats import mannwhitneyu
import pandas as pd

# Define function that performs Mann-Whitney U-testing for univariate feature selection
def mannwhitneyu_test(X_train, y_train):

    # Results for each feature
    p_values = []
    for col in X_train.columns:

        # Mann-Whitney U-test between 2 classes
        group1 = X_train[col][y_train == 'benign']  # Benign class
        group2 = X_train[col][y_train == 'malignant']  # Malignant class 
        _, p_value = mannwhitneyu(group1, group2)
        p_values.append(p_value)
        
    return pd.Series(p_values, index=X_train.columns)

# Calculating p-values
print("Running Mann-Whitney U test...")
p_values = mannwhitneyu_test(X_train, y_train)

# Select top 50 features with lowest p-Values (most signficant features)
best_features_mwu = p_values.nsmallest(50)
print(f"The 50 best features selected by Mann-Whitney U test are: {best_features_mwu.index.tolist()}")

# Select top 50 p-values (most signifcant features)
top_50_p_values = p_values.nsmallest(50)

# Print p-values of top 50 significant features
print("\nTop 50 p-values (lowest p-values):")
print(top_50_p_values.values)

significance_threshold = 0.05

# Identify signficant features based on p-value treshold
significant_features = p_values[p_values < significance_threshold]

# Print amount of significant features
print(f"Aantal significante features volgens de Mann-Whitney U-test: {len(significant_features)}")

# Print list of significant features
print(f"Significante features: {significant_features.index.tolist()}")