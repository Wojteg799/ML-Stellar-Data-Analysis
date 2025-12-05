# ML Stellar Data Analysis

Predictive modeling and exploratory analysis of star properties based on an astronomical dataset.  
The goal of this project is to explore relationships between stellar features (e.g. temperature, luminosity, radius, absolute magnitude) and build machine learning models that can predict or classify star characteristics.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Business / Scientific Motivation](#business--scientific-motivation)
- [Dataset](#dataset)
- [Objectives](#objectives)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

---

## Project Overview

This repository contains a complete end-to-end data project:

1. Exploratory Data Analysis of stellar properties.
2. Data cleaning and preprocessing
3. Feature engineering.
4. Model training and evaluation: building and comparing ML models to predict the star type.
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
   - Decision Tree
   - Random Forest
   - Neural network
4. Evaluate models using appropriate metrics:
   - Classification: accuracy, precision, recall, F1, confusion matrix.
5. Document the workflow in a clear and reproducible way.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ML-Stellar-Data-Analysis
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure data file is present:**
   - The dataset file `stars.csv` should be in `data/raw/` directory
   - If missing, download from [Kaggle](https://www.kaggle.com/datasets/deepu1109/star-dataset)

---

## Usage

### Running the Complete Pipeline

To run the entire analysis pipeline (preprocessing, visualization, modeling, and evaluation):

```bash
python src/main.py
```

This will:
1. Load and preprocess the data
2. Create all visualizations (saved to `outputs/`)
3. Train and optimize three models (Decision Tree, Random Forest, Neural Network)
4. Evaluate and compare models
5. Save trained models to `models/`
6. Generate all plots and visualizations in `outputs/`

### Using Individual Modules

You can also import and use individual modules in your own scripts:

```python
from src.data_preprocessing import preprocess_pipeline
from src.models import DecisionTreeModel, RandomForestModel, NeuralNetworkModel

# Preprocess data
data = preprocess_pipeline()

# Train a specific model
dt_model = DecisionTreeModel()
dt_model.optimize(data['X_train_scaled'], data['y_train'])
results = dt_model.evaluate(data['X_test_scaled'], data['y_test'])
```

### Jupyter Notebook

For interactive exploration, use the Jupyter notebook:
```bash
jupyter notebook notebooks/stellar_analisys_notebook.ipynb
```

---

## Results

The pipeline generates the following outputs in the `outputs/` directory:

**Visualizations:**
- `distributions.png` - Distribution of numeric variables
- `correlation_matrix.png` - Correlation heatmap
- `3d_scatter.html` - Interactive 3D scatter plot
- `hr_diagram.html` - Hertzsprung-Russell diagram
- `ternary_plot.html` - Ternary plot of star parameters

**Model Evaluation:**
- `decision_tree_visualization.png` - Decision tree structure
- `decision_tree_confusion_matrix.png` - Decision Tree confusion matrix
- `random_forest_confusion_matrix.png` - Random Forest confusion matrix
- `random_forest_pca_decision_boundary.png` - PCA decision boundary visualization
- `neural_network_confusion_matrix.png` - Neural Network confusion matrix
- `model_comparison_confusion_matrices.png` - Side-by-side comparison of all models

**Models:**
- `models/decision_tree_model.pkl` - Trained Decision Tree
- `models/random_forest_model.pkl` - Trained Random Forest
- `models/neural_network_model.pkl` - Trained Neural Network
- `models/scaler.pkl` - Feature scaler for preprocessing new data

**Expected Performance:**
Based on the analysis, Random Forest typically achieves the highest accuracy (~94-95%) for star type classification, followed by Neural Network (~91-92%) and Decision Tree (~90-91%).

---

## Tech Stack

**Language & Core Libraries:**
- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Plotly
- Kaleido (for static plot export)
- Joblib (for model serialization)

---

## Project Structure

This project follows a standard, production-like structure for data/ML work:

```bash
ML-Stellar-Data-Analysis/
│
├── README.md                    # Project description (this file)
├── requirements.txt             # Python dependencies
│
├── data/
│   ├── raw/                     # Original, immutable data
│   │   └── stars.csv            # Raw star dataset
│   └── processed/                # Processed data (generated)
│
├── notebooks/      
│   ├── stellar_analisys_notebook.ipynb    # Jupyter notebook with full analysis
│   └── stellar_analisys_notebook.py       # Python version of notebook
│
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── config.py                 # Configuration and constants
│   ├── data_preprocessing.py     # Data loading, cleaning, and preprocessing
│   ├── visualization.py          # All visualization functions
│   ├── main.py                   # Main script to run complete pipeline
│   └── models/                   # ML model implementations
│       ├── __init__.py
│       ├── decision_tree.py      # Decision Tree model
│       ├── random_forest.py      # Random Forest model
│       └── neural_network.py     # Neural Network model
│
├── models/                       # Saved trained models (generated)
│
└── outputs/                      # Generated outputs (plots, visualizations)
```

---
