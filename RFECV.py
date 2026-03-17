#load data
from assignment import X_train, y_train

#imports
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import feature_selection, model_selection
import matplotlib.pyplot as plt
import sklearn.datasets as ds
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split 
import seaborn as sns
import matplotlib.pyplot as plt

from worcliver.load_data import load_data



# Create the RFE object and compute a cross-validated score.
svc = svm.SVC(kernel="rbf")

from sklearn.ensemble import RandomForestClassifier

#classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rfecv = feature_selection.RFECV(
    estimator=rf, step=1,
    cv=model_selection.StratifiedKFold(4),
    scoring='roc_auc')
rfecv.fit(X_train, y_train)
print("RFECV fitting complete.")


# Vind de index van de iteratie waar 50 features geselecteerd zijn
features_50_index = np.where(np.array(range(1, len(rfecv.cv_results_["mean_test_score"]) + 1)) == 50)[0][0]

# Haal de score voor die iteratie
score_50_features = rfecv.cv_results_["mean_test_score"][features_50_index]
print(f"Cross-validation score for 50 selected features: {score_50_features}")

# Haal de 50 geselecteerde features op
selected_50_features = X_train.columns[rfecv.support_][:50]
print(f"The 50 selected features are: {selected_50_features}")

print(len(selected_50_features))

# Plot de cross-validatie scores per aantal geselecteerde features
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.cv_results_["mean_test_score"]) + 1), rfecv.cv_results_["mean_test_score"])
plt.show()