#%% Main code Liver assignment 
from worcliver.load_data import load_data
data = load_data()

print("data geladen")

X_train = pd.read_pickle("X_train_filtered.pkl")
X_test = pd.read_pickle("X_test_filtered.pkl")
y_train = pd.read_pickle("y_train.pkl")
y_test = pd.read_pickle("y_test.pkl")

     
