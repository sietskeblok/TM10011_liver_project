print("script gestart")

from worcliver.load_data import load_data

print("voor load_data")
data = load_data()
print("na load_data")
print(data.shape)