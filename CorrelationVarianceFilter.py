# This script performs variance and correlation filtering on the dataset and provides insight in remaining features

# Import modules
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
warnings.filterwarnings("ignore") 

# Correlation filter to remove highly correlated features
class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):

        # Set correlation treshold above which a feature will be removed
        self.threshold = threshold
        self.to_drop_ = None
        self.columns_ = None

    def fit(self, X, y=None):

        # Convert input to dataframe 
        X = pd.DataFrame(X)
        self.columns_ = X.columns

        # Compute absolute correlation matrix
        corr_matrix = X.corr().abs()

        # Keep upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Identify features to remove that have a correlation above treshold
        self.to_drop_ = [col for col in upper.columns if any(upper[col] > self.threshold)]

        return self

    def transform(self, X):

        # Convert input to dataframe
        X = pd.DataFrame(X)
        X.columns = self.columns_

        # Remove highly correlated columns
        return X.drop(columns=self.to_drop_, errors='ignore')

# Load pre-processed training and test data
X_train = pd.read_pickle("X_train.pkl")
X_test = pd.read_pickle("X_test.pkl")
y_train = pd.read_pickle("y_train.pkl")
y_test = pd.read_pickle("y_test.pkl")

# Apply variance filter
var_filter = VarianceThreshold(threshold=0.0)
var_filter.fit(X_train)
X_filtered_var = var_filter.transform(X_train)

# Apply correlation filter
corr_filter = CorrelationFilter(threshold=0.95)
corr_filter.fit(X_filtered_var)
X_filtered_cor = corr_filter.transform(X_filtered_var)

# Print shape of data to observe feature extraction
print(f"The initial dataset consists of {X_train.shape[0]} samples and {X_train.shape[1]} features.")
print(f"After variance filtering: {X_filtered_var.shape[0]} samples and {X_filtered_var.shape[1]} features remain.")
print(f"After correlation filtering: {X_filtered_cor.shape[0]} samples and {X_filtered_cor.shape[1]} features remain.")

