from worcliver.load_data import load_data
data = load_data()

# DATA SHAPE bekijken
print(data.shape)
# Check of er überhaupt missende waarden zijn
print(data.isnull().values.any())

# OUTLIERS bekijken
from assignment import X
from assignment import y

# Alleen kolommen met getallen erin, 
X_num = X.select_dtypes(include='number')
# Bereken Q1 en Q3
Q1 = X_num.quantile(0.25)
Q3 = X_num.quantile(0.75)

# Bereken IQR
IQR = Q3 - Q1
# Definieer grenzen
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identificeer in hoeveel samples outliers zitten
outliers = data[(X < lower_bound) | (X > upper_bound)]
# Print het aantal outliers
print(f"Aantal outliers: {len(outliers)}")

# Identificeer totaal aantal outliers
outliers = (X_num < lower_bound) | (X_num > upper_bound)
print(f"Aantal outliers totaal: {outliers.sum().sum()}")

print(X.dtypes)

