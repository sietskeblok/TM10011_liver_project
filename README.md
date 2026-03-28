# Liver Tumour Classification using Machine Learning

This repository contains the project of Group 7 for the course Machine Learning (TM10011.
The objective of this assignment is to develop a machine learning model capable of distinguishing benign from malignant primary solid liver tumours using radiomic features extracted from T2-weighted MRI scans.

# Project Overview

The goal of this project was to build a classification model that predicts whether a liver lesion is benign or malignant.
Multiple feature selection methods and classification models were evaluated and compared using nested cross-validation.
The best-performing model was a Random Forest classifier without prior feature selection, achieving a ROC AUC of 0.76 on the test set.

This analysis was performed using Python version 3.14.2, numpy 2.4.2, pandas 3.0.1, scikit-learn 1.8.0, seaborn 0.13.2 and matplotlib 3.10.8.

# Dataset

The dataset consists of radiomic features extracted from T2-weighted MRI scans of liver lesions.
Each observation represents a liver lesion and contains:
- Radiomic imaging features
- A label:
  - benign (0)
  - malignant (1)

The dataset was split into:
- Training set (used for model development and nested cross-validation)
- Test set (held out for final evaluation)

# Machine Learning Pipeline

The modelling pipeline consisted of the following steps:

1. Preprocessing
   - Variance filtering
   - Correlation filtering
   - Robust scaling

2. Feature Selection
   - Mann–Whitney U test
   - RFECV
   - No feature selection

3. Classification Models
   - Logistic Regression
   - Support Vector Machine (SVM)
   - Random Forest

5. Model Selection
   - Hyperparameter tuning using GridSearchCV
   - Nested cross-validation to prevent data leakage

# Results

Best model configuration:
  - Classifier: Random Forest
  - Feature Selection: None
  - Evaluation metric: ROC AUC
  - Test ROC AUC: 0.76

# Repository Structure

Main scripts in this repository:
  - ImportData.py
  Loads the dataset and performs the initial train–test split.
  - FinalModel.py
  Runs feature selection, hyperparameter tuning, model training, and evaluation.

Remaining scripts:
   - CorelationVarianceFilter.py
   Performs variance and correlation filtering on dataset and provides insight in remaining features
   - Data_inspection.py
   Inspects the data 
   - Normal_distribution.py
   Assesses normal distribution of each feature in dataset 
   - RFECV.py
   Feature selection RFECV with Random Forest to select k-best features

# How to Run the Project

To reproduce the results, run the scripts in the following order:

1. ImportData.py
2. FinalModel.py

# Authors

Group 7: Sietske Blok (S.J.R.Blok@student.tudelft.nl), Donna de Leur (D.deLeur@student.tudelft.nl), Piotr van Dijk (P.S.vanDijk@student.tudelft.nl) and Max Lindaart (M.H.Lindaart@student.tudelft.nl).

