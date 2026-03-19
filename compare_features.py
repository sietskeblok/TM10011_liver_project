from MannwhitneyU import significant_features
from RFECV import selected_50_features

set_1 = set(significant_features.index.tolist())
set_2 = set(selected_50_features)

# Vind de gemeenschappelijke features tussen de twee sets
common_features = set_1.intersection(set_2)

# Print het aantal gemeenschappelijke features
print(f"Aantal gemeenschappelijke features: {len(common_features)}")

# Print de lijst van gemeenschappelijke features
print(f"Gemeenschappelijke features: {common_features}")