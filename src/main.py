"""
Main script for Stellar Data Analysis project.
Runs the complete pipeline: preprocessing, visualization, modeling, and evaluation.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import preprocess_pipeline
from src.visualization import create_all_visualizations, plot_model_comparison
from src.models import DecisionTreeModel, RandomForestModel, NeuralNetworkModel
from src.config import NUMERIC_COLUMNS, OUTPUTS_DIR, MODELS_DIR
import joblib


def main():
    """Main execution function."""
    print("="*70)
    print("STELLAR DATA ANALYSIS - COMPLETE PIPELINE")
    print("="*70)
    
    # Step 1: Preprocessing
    print("\n" + "="*70)
    print("STEP 1: DATA PREPROCESSING")
    print("="*70)
    
    data = preprocess_pipeline(save_processed=True)
    
    # Step 2: Visualization
    print("\n" + "="*70)
    print("STEP 2: DATA VISUALIZATION")
    print("="*70)
    
    create_all_visualizations(data['df'], save_to_outputs=True)
    
    # Step 3: Model Training and Evaluation
    print("\n" + "="*70)
    print("STEP 3: MODEL TRAINING AND EVALUATION")
    print("="*70)
    
    # Decision Tree
    dt_model = DecisionTreeModel()
    dt_model.train_baseline(data['X_train_scaled'], data['y_train'])
    dt_baseline_results = dt_model.evaluate(
        data['X_test_scaled'], 
        data['y_test'], 
        use_optimized=False,
        save_plots=False
    )
    
    dt_model.optimize(data['X_train_scaled'], data['y_train'])
    dt_results = dt_model.evaluate(
        data['X_test_scaled'], 
        data['y_test'], 
        use_optimized=True,
        save_plots=True
    )
    
    dt_model.visualize_tree(NUMERIC_COLUMNS)
    
    print(f"\nDecision Tree improvement: {dt_results['accuracy'] - dt_baseline_results['accuracy']:.4f}")
    
    # Random Forest
    rf_model = RandomForestModel()
    rf_model.train_baseline(data['X_train_scaled'], data['y_train'])
    rf_baseline_results = rf_model.evaluate(
        data['X_test_scaled'], 
        data['y_test'], 
        use_optimized=False,
        save_plots=False
    )
    
    rf_model.optimize(data['X_train_scaled'], data['y_train'])
    rf_results = rf_model.evaluate(
        data['X_test_scaled'], 
        data['y_test'], 
        use_optimized=True,
        save_plots=True
    )
    
    rf_model.visualize_pca_decision_boundary(
        data['X_train_scaled'], 
        data['y_train']
    )
    
    print(f"\nRandom Forest improvement: {rf_results['accuracy'] - rf_baseline_results['accuracy']:.4f}")
    
    # Neural Network
    nn_model = NeuralNetworkModel()
    nn_model.train_baseline(data['X_train_scaled'], data['y_train'])
    nn_baseline_results = nn_model.evaluate(
        data['X_test_scaled'], 
        data['y_test'], 
        use_optimized=False,
        save_plots=False
    )
    
    nn_model.optimize(data['X_train_scaled'], data['y_train'])
    nn_results = nn_model.evaluate(
        data['X_test_scaled'], 
        data['y_test'], 
        use_optimized=True,
        save_plots=True
    )
    
    print(f"\nNeural Network improvement: {nn_results['accuracy'] - nn_baseline_results['accuracy']:.4f}")
    
    # Step 4: Model Comparison
    print("\n" + "="*70)
    print("STEP 4: MODEL COMPARISON")
    print("="*70)
    
    predictions_dict = {
        'Decision Tree': dt_results['predictions'],
        'Random Forest': rf_results['predictions'],
        'Neural Network': nn_results['predictions']
    }
    
    plot_model_comparison(data['y_test'], predictions_dict)
    
    # Print summary
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"Decision Tree Accuracy:     {dt_results['accuracy']:.4f}")
    print(f"Random Forest Accuracy:     {rf_results['accuracy']:.4f}")
    print(f"Neural Network Accuracy:    {nn_results['accuracy']:.4f}")
    
    accuracies = {
        'Decision Tree': dt_results['accuracy'],
        'Random Forest': rf_results['accuracy'],
        'Neural Network': nn_results['accuracy']
    }
    best_model_name = max(accuracies, key=accuracies.get)
    best_accuracy = accuracies[best_model_name]
    
    print(f"\nBest Model: {best_model_name} with accuracy {best_accuracy:.4f}")
    
    # Save models
    print("\n" + "="*70)
    print("SAVING MODELS")
    print("="*70)
    
    joblib.dump(dt_model.optimized_model, MODELS_DIR / "decision_tree_model.pkl")
    joblib.dump(rf_model.optimized_model, MODELS_DIR / "random_forest_model.pkl")
    joblib.dump(nn_model.optimized_model, MODELS_DIR / "neural_network_model.pkl")
    joblib.dump(data['scaler'], MODELS_DIR / "scaler.pkl")
    
    print(f"Models saved to {MODELS_DIR}")
    print(f"\nAll outputs saved to {OUTPUTS_DIR}")
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()

