# ML Stellar Data Analysis

Predictive modeling and exploratory analysis of star properties based on an astronomical dataset.  
The goal of this project is to explore relationships between stellar features (e.g. temperature, luminosity, radius, absolute magnitude) and build machine learning models that can predict or classify star characteristics.

---

## Table of Contents

TBD

---

## Project Overview

This repository contains a complete end-to-end data project:

1. Exploratory Data Analysis of stellar properties.
2. Data cleaning and preprocessing: handling missing values, outliers, and inconsistent records.
3. Feature engineering: transforming raw physical measurements into features suitable for machine learning.
4. Model training and evaluation: building and comparing ML models to predict a chosen target (e.g. star type / class or a continuous property such as absolute magnitude).
5. Reproducible project structure: separation of raw data, processed data, notebooks, and production-ready Python modules in `src/`.

The project is designed to mimic a real-world data science / ML workflow that i want to use for my portfolio

---

## Business / Scientific Motivation

Understanding relationships between stellar properties is a core topic in astrophysics (e.g. Hertzsprung–Russell diagram).  
From a data perspective, this problem is a good playground to:

- Work with continuous physical variables
- Explore correlations and non-linear relationships between features.
- Practice regression or classification ML tasks on a clean but non-trivial dataset.
- Demonstrate skills in data preprocessing, modeling, and evaluation in a scientific-style context.

---

## Dataset

- Name: Star dataset to predict star types
- Source: https://www.kaggle.com/datasets/deepu1109/star-dataset
- Number of rows: 240
- Number of features: 7 

This CSV file contains a dataset of 240 stars of 6 classes:

Brown Dwarf -> Star Type = 0

Red Dwarf -> Star Type = 1

White Dwarf-> Star Type = 2

Main Sequence -> Star Type = 3

Supergiant -> Star Type = 4

Hypergiant -> Star Type = 5

The Luminosity and radius of each star is calculated w.r.t. that of the values of Sun.
Lo = 3.828 x 10^26 Watts
Ro = 6.9551 x 10^8 m

Target definition:
- Classification: Predict star type / class.  
---

## Objectives

Main objectives of this project:

1. Explore the dataset and understand distributions and relationships between stellar features.
2. Preprocess the data to make it suitable for ML models:
   - Handle missing values (if any).
   - Encode categorical variables.
   - Scale / transform numeric features where appropriate.
3. Build and compare ML models, such as:
   - Logistic Regression / Linear Regression
   - Decision Tree
   - Random Forest
   - Gradient Boosting (optional)
4. Evaluate models using appropriate metrics:
   - Classification: accuracy, precision, recall, F1, confusion matrix.
   - Regression: MAE, RMSE, R².
5. Document the workflow in a clear and reproducible way.

---

## Tech Stack

Language & Core Libraries
- Python 3.x
- NumPy
- Pandas
- Matplotlib / Seaborn
- Scikit-learn
- Jupyter / IPython Notebooks
---

## Project Structure

This project follows a standard, production-like structure for data/ML work:

```bash
ML-Stellar-Data-Analysis/
│
├── README.md               # Project description (this file)
├── requirements.txt        # Python dependencies
├── .gitignore
│
├── data/
│   ├── raw/                # Original, immutable data dump
│   └── processed/          # Cleaned data, ready for modeling
│
│
├── notebooks/      
│   └── TBA                 # To be Added
│
├── src/     
│   └── TBA                 #To be added
│
├── models/                 # Saved models (e.g. model.pkl)
│
└── outputs/               #Outputs


