# Usage Examples

## Quick Start

Run the complete pipeline:
```bash
python src/main.py
```

## Using Individual Components

### 1. Data Preprocessing Only

```python
from src.data_preprocessing import preprocess_pipeline

# Run complete preprocessing
data = preprocess_pipeline(save_processed=True)

# Access preprocessed data
X_train = data['X_train_scaled']
X_test = data['X_test_scaled']
y_train = data['y_train']
y_test = data['y_test']
```

### 2. Create Visualizations Only

```python
from src.data_preprocessing import load_data, clean_data
from src.visualization import create_all_visualizations

# Load and clean data
df = load_data()
df = clean_data(df)

# Create all visualizations
create_all_visualizations(df, save_to_outputs=True)
```

### 3. Train a Single Model

```python
from src.data_preprocessing import preprocess_pipeline
from src.models import DecisionTreeModel

# Preprocess data
data = preprocess_pipeline()

# Train and evaluate Decision Tree
dt_model = DecisionTreeModel()
dt_model.optimize(data['X_train_scaled'], data['y_train'])
results = dt_model.evaluate(data['X_test_scaled'], data['y_test'])

print(f"Accuracy: {results['accuracy']:.4f}")
```

### 4. Compare Models Manually

```python
from src.data_preprocessing import preprocess_pipeline
from src.models import DecisionTreeModel, RandomForestModel, NeuralNetworkModel

# Preprocess data
data = preprocess_pipeline()

# Train all models
models = {
    'Decision Tree': DecisionTreeModel(),
    'Random Forest': RandomForestModel(),
    'Neural Network': NeuralNetworkModel()
}

results = {}
for name, model in models.items():
    model.optimize(data['X_train_scaled'], data['y_train'])
    results[name] = model.evaluate(data['X_test_scaled'], data['y_test'])

# Print comparison
for name, result in results.items():
    print(f"{name}: {result['accuracy']:.4f}")
```

### 5. Load Saved Model for Predictions

```python
import joblib
import numpy as np
from src.config import MODELS_DIR

# Load model and scaler
model = joblib.load(MODELS_DIR / "random_forest_model.pkl")
scaler = joblib.load(MODELS_DIR / "scaler.pkl")

# Prepare new data (example)
new_data = np.array([[3000, 0.01, 0.1, 15.0]])  # Temperature, Luminosity, Radius, Magnitude

# Scale and predict
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)

print(f"Predicted star type: {prediction[0]}")
```

