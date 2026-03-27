# This script reads the data and functions as its inspection

from worcliver.load_data import load_data
data = load_data()

# Data shape evaluation
print(data.shape)

# Checking for missing data
print(data.isnull().values.any())

# Evaluation outliers
from assignment import X
from assignment import y

# Evaluation of columns with mere integers
X_num = X.select_dtypes(include='number')

# Calculation of Q1 and Q3
Q1 = X_num.quantile(0.25)
Q3 = X_num.quantile(0.75)

# Calculation of IQR
IQR = Q3 - Q1

# Define upper and lower bound
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identification of amount of samples including outliers
outliers = data[(X < lower_bound) | (X > upper_bound)]

# Print amount of outliers
print(f"Aantal outliers: {len(outliers)}")

# Identify total number of outliers
outliers = (X_num < lower_bound) | (X_num > upper_bound)
print(f"Aantal outliers totaal: {outliers.sum().sum()}")

print(X.dtypes)

