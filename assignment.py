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

# Check of er überhaupt missende waarden zijn
print(data.isnull().values.any())


from sklearn import svm
from sklearn import feature_selection, model_selection
import matplotlib.pyplot as plt
import sklearn.datasets as ds

from sklearn.ensemble import RandomForestClassifier



# Create the RFE object and compute a cross-validated score.
svc = svm.SVC(kernel="rbf")


from sklearn.ensemble import RandomForestClassifier

#classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rfecv = feature_selection.RFECV(
    estimator=rf, step=5,
    cv=model_selection.StratifiedKFold(4),
    scoring='roc_auc')
rfecv.fit(X_train, y_train)


# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.cv_results_["mean_test_score"]) + 1), rfecv.cv_results_["mean_test_score"])
plt.show()
