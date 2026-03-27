# This script assesses the normal distribution of each feature in the dataset 

# Import modules
from ImportData import X, y

# Shapiro-Wilk test for normal distribution 
from scipy.stats import shapiro

# Creation of lists to save Shapiro-Wilk test results
normal_features_sw = []
non_normal_features_sw = []

# Assessment of Shapiro-Wilk test for each feature
for column in X.columns:
    stat, p_value = shapiro(X[column].dropna())

    # Non-normal distribution if p < 0.05 
    if p_value < 0.05:
        non_normal_features_sw.append(column)
    else:
        normal_features_sw.append(column)

# Calculation of percentages
total_features = len(X.columns)
normal_percentage = (len(normal_features_sw) / total_features) * 100
non_normal_percentage = (len(non_normal_features_sw) / total_features) * 100

# Print results
print(f"Amount of normally distributed features assessed with Shapiro-Wilk test: {len(normal_features_sw)} ({normal_percentage:.2f}%)")
print(f"Amount of non-normally distributed features assessed with Shapiro-Wilk test: {len(non_normal_features_sw)} ({non_normal_percentage:.2f}%)")