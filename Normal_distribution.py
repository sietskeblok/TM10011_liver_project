# controleren normale verdeling data-set met Anderson-Darrling test
from scipy.stats import anderson

# Import de variabelen uit assignment.py
from assignment import X_train, y_train


# Lijst om Anderson-Darling resultaten op te slaan
normal_features = []
non_normal_features = []

# Voer de Anderson-Darling test uit voor elke feature
for column in X_train.columns:
    result = anderson(X_train[column].dropna(), dist='norm')
    
    # Als de teststatistiek groter is dan de kritische waarde bij 5% significatie, dan is het niet normaal verdeeld
    if result.statistic > result.critical_values[2]:  # Vergelijk met de kritische waarde voor 5% significantie
        non_normal_features.append(column)
    else:
        normal_features.append(column)

# Bereken het percentage van normaal en niet-normaal verdeelde features
total_features = len(X_train.columns)
normal_percentage = (len(normal_features) / total_features) * 100
non_normal_percentage = (len(non_normal_features) / total_features) * 100

# Print de resultaten
print(f"Aantal normaal verdeelde features: {len(normal_features)} ({normal_percentage:.2f}%)")
print(f"Aantal niet normaal verdeelde features: {len(non_normal_features)} ({non_normal_percentage:.2f}%)")
