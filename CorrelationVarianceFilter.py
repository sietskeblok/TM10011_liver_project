import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold

from FeatureselModel import X_filtered

warnings.filterwarnings("ignore") 


#Correlation filter definition
class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.to_drop_ = None
        self.columns_ = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = X.columns

        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop_ = [col for col in upper.columns if any(upper[col] > self.threshold)]

        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        X.columns = self.columns_
        return X.drop(columns=self.to_drop_, errors='ignore')

#Data reading
X_train = pd.read_pickle("X_train.pkl")
X_test = pd.read_pickle("X_test.pkl")
y_train = pd.read_pickle("y_train.pkl")
y_test = pd.read_pickle("y_test.pkl")


var_filter = VarianceThreshold(threshold=0.0)
var_filter.fit(X_train)
X_filtered_var = var_filter.transform(X_train)


corr_filter = CorrelationFilter(threshold=0.95)
corr_filter.fit(X_filtered_var)
X_filtered_cor = corr_filter.transform(X_filtered_var)


print(X_train.shape)
print(X_filtered_var.shape)
print(X_filtered_cor.shape)

