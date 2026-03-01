# ## Data loading and cleaning
# Below are functions to load the dataset of your choice. 
# After that, it is all up to you to create and evaluate a classification method. 
# Beware, there may be missing values in these datasets. 
# Good luck!


#%% Data loading functions. Uncomment the one you want to use
from worcliver.load_data import load_data

data = load_data()
print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')
print("Hopelijk werkt dit")

# Aantal missende waarden per kolom
print(data.isnull().sum())

# Check of er überhaupt missende waarden zijn
print(data.isnull().values.any())