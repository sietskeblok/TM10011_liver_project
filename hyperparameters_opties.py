'''
This module defines the hyperparameters for the SVM model and the grid search.
We kunnen er nog voor kiezen een polynomial kernel toe te voegen, maar dat is wel computationally expensive. Lees ook dat het meestal niet gebruikt wordt in de praktijk, dus ik heb het er nog niet in gezet. Heb daar neit echt een bron voor ook ben ik bang.
param_grid = [
    {
        "svm__kernel": ["linear"],
        "svm__C": [0.01, 0.1, 1, 10, 100]
    },
    {
        "svm__kernel": ["rbf"],
        "svm__C": [0.01, 0.1, 1, 10, 100],
        "svm__gamma": ["scale", 0.001, 0.01, 0.1]
    }
]

'''

'''
Dit is voor de LogReg. 
param_grid = {
    'logreg__penalty': ['l1', 'l2'],
    'logreg__C': [0.01, 0.1, 1, 10, 100],
    'logreg__solver': ['liblinear'],  # because we want L1 support
}
'''
'''
param_grid = {
    'rf__n_estimators': [200, 500],
    'rf__max_depth': [None, 10, 20],
    'rf__min_samples_split': [2, 5],
    'rf__min_samples_leaf': [1, 2, 5],
    'rf__max_features': ['sqrt', 'log2'],
    'rf__class_weight': [None, 'balanced']
}

Hier kunnen ook nog dingen aan toegevoegd worden, zoals max_samples, max_leaf_nodes, min_weight_fraction_leaf en de criterion, maar ik denk dat dit ook prima is.




'''
