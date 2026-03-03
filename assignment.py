# ## Data loading and cleaning
# Below are functions to load the dataset of your choice. 
# After that, it is all up to you to create and evaluate a classification method. 
# Beware, there may be missing values in these datasets. 
# Good luck!


#%% Data loading 
from sklearn.linear_model import LogisticRegression

from worcliver.load_data import load_data

data = load_data()
print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')

benigne_count = data[data['label'] == 'benign'].shape[0]
maligne_count = data[data['label'] == 'malignant'].shape[0]

print(data['label'].unique())

print(f"Aantal samples met label 'benign': {benigne_count}")
print(f"Aantal samples met label 'malignant': {maligne_count}")

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split 

import seaborn as sns
import matplotlib.pyplot as plt
#print(data.columns)



# Verwijder label uit kolommen voor alleen features 
X = data.drop(columns=['label'])  # Verwijder zowel 'label' als 'ID' van de features
y = data['label']  # De targetvariabele is 'label'

from scipy.stats import anderson
# Lijst om Anderson-Darling resultaten op te slaan

normal_features = []
non_normal_features = []

# Voer de Anderson-Darling test uit voor elke feature
for column in X.columns:
    result = anderson(X[column].dropna(), dist='norm')
    
    # Als de teststatistiek groter is dan de kritische waarde bij 5% significatie, dan is het niet normaal verdeeld
    if result.statistic > result.critical_values[2]:  # Vergelijk met de kritische waarde voor 5% significantie
        non_normal_features.append(column)
    else:
        normal_features.append(column)

# Bereken het percentage van normaal en niet-normaal verdeelde features
total_features = len(X.columns)
normal_percentage = (len(normal_features) / total_features) * 100
non_normal_percentage = (len(non_normal_features) / total_features) * 100

# Print de resultaten
print(f"Aantal normaal verdeelde features: {len(normal_features)} ({normal_percentage:.2f}%)")
print(f"Aantal niet normaal verdeelde features: {len(non_normal_features)} ({non_normal_percentage:.2f}%)")

# Bereken de eerste (Q1) en derde kwartielen (Q3)
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)

# Bereken de IQR
IQR = Q3 - Q1

# Definieer de grenzen voor outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identificeer outliers
outliers = data[(X < lower_bound) | (X > upper_bound)]

# Print het aantal outliers
print(f"Aantal outliers: {len(outliers)}")

# Splits de data in trainings- en testsets (bijv. 80% training en 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f'Training set size: {len(X_train)}')
print(f'Test set size: {len(X_test)}')

# scaling 
from sklearn.preprocessing import RobustScaler

# Maak een instantie van de RobustScaler
scaler = RobustScaler()

# Pas de scaler toe op de trainingsdata
X_scaled = scaler.fit_transform(X_train)

import seaborn as sns
import matplotlib.pyplot as plt

# Visualiseer een boxplot van een specifieke feature na scaling
sns.boxplot(x=X_scaled[:, 3])  # Kies bijvoorbeeld de eerste feature
plt.title('Boxplot van de eerste feature na Robust Scaling')
plt.show()

# fearure selection alleen op trainingsdata om overfitting te voorkomen
## Bereken de correlatiematrix tussen de features
 #corr_matrix = X.corr()

## Maak een heatmap van de correlatiematrix
#plt.figure(figsize=(12, 8))
#sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar_kws={'shrink': 0.8})
#plt.title('Correlation Matrix of Features')
#plt.show()

# Stel Stratified K-Fold in met 5 vouwen
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Voorbeeld van het trainen van een RandomForest en cross-validatie
model = LogisticRegression()

# Gebruik cross_val_score om het model te evalueren met 5-voudige cross-validatie
scores = cross_val_score(model, X, y, cv=skf)

# Print de prestaties voor elke fold
print(f"Scores per fold: {scores}")
# Gemiddelde nauwkeurigheid
print(f"Gemiddelde nauwkeurigheid over 5 folds: {scores.mean()}")


# %%
