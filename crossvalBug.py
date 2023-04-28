# File        :   crossvalBug.py
# Version     :   1.0.0
# Description :   Minimal reproducible example for cross_val_score crashing
#                
# Date:       :   Apr 27, 2023

import os
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVC

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

# Dataset path:
rootDir = "D:"
baseDir = "dataSets"
fileName = "dataset.csv"

# Create file path:
path = os.path.join(rootDir, baseDir, fileName)

# Load dataset:
df_train = pd.read_csv(path)

# Get training features and label:
y_train = df_train["target"]
X_train = df_train.drop("target", axis=1)

# Build the classifier pipeline:
classifierPipeline = make_pipeline(PolynomialFeatures(degree=2),
                                   StandardScaler(),
                                   LinearSVC(C=1, random_state=42, dual=False))

# Pipeline Fit:
print("Fitting pipeline...")
classifierPipeline.fit(X_train, y_train)

# Cross-validation setup:
cvFolds = 5
parallelJobs = 5

# Check out the classifier accuracy using cross-validation:
print("Running cross-validation...")
classifierAccuracy = cross_val_score(estimator=classifierPipeline, X=X_train, y=y_train, cv=cvFolds,
                                     n_jobs=parallelJobs, verbose=3)

# Accuracy for each fold:
print("Cross Score:", classifierAccuracy)
