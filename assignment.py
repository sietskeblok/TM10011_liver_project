#%% Main code Liver assignment 

#run dit eerst
from worcliver.load_data import load_data
import pandas as pd

data = load_data()
#omzetten naar pickle zodat het ingelezen kan worden in andere files
data.to_pickle("data.pkl")
print("data geladen")

#run hierna pre-processing.py
#run daarna FeatureselModel.py


