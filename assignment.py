#%% Main code Liver assignment 
from worcliver.load_data import load_data
import pandas as pd

data = load_data()
#omzetten naar pickle zodat het ingelezen kan worden in andere files
data.to_pickle("data.pkl")
print("data geladen")

#run hierna pre-processing.py

#plak wat hieronder staat vervolgens bovenaan de andere code die je wil runnen
'''
X_train = pd.read_pickle("X_train_filtered_scaled.pkl")
X_test = pd.read_pickle("X_test_filtered_scaled.pkl")
y_train = pd.read_pickle("y_train.pkl")
y_test = pd.read_pickle("y_test.pkl")
''' 
