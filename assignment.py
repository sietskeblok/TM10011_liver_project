#%% Main code Liver assignment 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split 
import seaborn as sns
import matplotlib.pyplot as plt


from worcliver.load_data import load_data
data = load_data()

# Verwijder label uit kolommen voor alleen features 
X = data.drop(columns=['label'])  # Verwijder zowel 'label' als 'ID' van de features
y = data['label']  # De targetvariabele is 'label'

# Splits de data in trainings- en testsets (bijv. 80% training en 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


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


# scaling 
from sklearn.preprocessing import RobustScaler

# Maak een instantie van de RobustScaler
scaler = RobustScaler()

# Pas de scaler toe op de trainingsdata
X_scaled = scaler.fit_transform(X_train)

import seaborn as sns
import matplotlib.pyplot as plt

# Visualiseer een boxplot van een specifieke feature na scaling
sns.boxplot(x=X_scaled[:, 80])  # Kies bijvoorbeeld de eerste feature
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
scores = cross_val_score(model, X_train, y_train, cv=skf)

# Print de prestaties voor elke fold
print(f"Scores per fold: {scores}")
# Gemiddelde nauwkeurigheid
print(f"Gemiddelde nauwkeurigheid over 5 folds: {scores.mean()}")


# %%
print("Hopelijk werkt dit")

# Aantal missende waarden per kolom
print(data.isnull().sum())

# Check of er überhaupt missende waarden zijn
print(data.isnull().values.any())