"""
Configuration file for the Stellar Data Analysis project.
Contains paths, constants, and hyperparameters.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
OUTPUTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
DATA_PROCESSED.mkdir(exist_ok=True)

# Data file
STARS_CSV = DATA_RAW / "stars.csv"

# Numeric columns for modeling
NUMERIC_COLUMNS = ['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)', 'Absolute magnitude(Mv)']

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.8

# Decision Tree hyperparameters for GridSearch
DT_PARAMS = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Random Forest hyperparameters for GridSearch
RF_PARAMS = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Neural Network hyperparameters for GridSearch
NN_PARAMS = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01]
}

# Cross-validation settings
CV_FOLDS = 5
SCORING = 'accuracy'

# Neural Network settings
NN_MAX_ITER = 1000

