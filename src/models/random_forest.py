"""
Random Forest model for star type classification.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

from src.config import RF_PARAMS, RANDOM_STATE, CV_FOLDS, SCORING, OUTPUTS_DIR, NUMERIC_COLUMNS
from src.visualization import plot_confusion_matrix, plot_pca_decision_boundary


class RandomForestModel:
    """Random Forest classifier with optimization capabilities."""
    
    def __init__(self, random_state=None):
        """
        Initialize Random Forest model.
        
        Args:
            random_state: Random state for reproducibility
        """
        if random_state is None:
            random_state = RANDOM_STATE
        
        self.random_state = random_state
        self.model = None
        self.optimized_model = None
        self.best_params = None
        
    def train_baseline(self, X_train, y_train):
        """
        Train baseline model with default parameters.
        
        Args:
            X_train: Training features
            y_train: Training labels
        
        Returns:
            Trained model
        """
        print("\n" + "="*50)
        print("RANDOM FOREST - BASELINE MODEL")
        print("="*50)
        
        self.model = RandomForestClassifier(random_state=self.random_state)
        self.model.fit(X_train, y_train)
        
        return self.model
    
    def optimize(self, X_train, y_train, param_grid=None, cv=None, scoring=None):
        """
        Optimize hyperparameters using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Parameter grid for search. If None, uses config default.
            cv: Number of CV folds. If None, uses config default.
            scoring: Scoring metric. If None, uses config default.
        
        Returns:
            Best estimator
        """
        if param_grid is None:
            param_grid = RF_PARAMS
        if cv is None:
            cv = CV_FOLDS
        if scoring is None:
            scoring = SCORING
        
        print("\n" + "="*50)
        print("RANDOM FOREST - HYPERPARAMETER OPTIMIZATION")
        print("="*50)
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=self.random_state),
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        self.optimized_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        print(f"Best parameters: {self.best_params}")
        
        return self.optimized_model
    
    def predict(self, X, use_optimized=True):
        """
        Make predictions.
        
        Args:
            X: Features
            use_optimized: Whether to use optimized model
        
        Returns:
            Predictions
        """
        model = self.optimized_model if use_optimized and self.optimized_model is not None else self.model
        
        if model is None:
            raise ValueError("Model not trained yet. Call train_baseline() or optimize() first.")
        
        return model.predict(X)
    
    def evaluate(self, X_test, y_test, use_optimized=True, save_plots=True):
        """
        Evaluate model and print metrics.
        
        Args:
            X_test: Test features
            y_test: Test labels
            use_optimized: Whether to use optimized model
            save_plots: Whether to save visualization plots
        
        Returns:
            Dictionary with evaluation metrics
        """
        model = self.optimized_model if use_optimized and self.optimized_model is not None else self.model
        model_name = "Optimized" if use_optimized and self.optimized_model is not None else "Baseline"
        
        if model is None:
            raise ValueError("Model not trained yet. Call train_baseline() or optimize() first.")
        
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"\n{model_name} Random Forest Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        
        if save_plots and use_optimized:
            # Save confusion matrix
            plot_confusion_matrix(
                y_test, 
                predictions, 
                f"Random Forest - {model_name}",
                OUTPUTS_DIR / "random_forest_confusion_matrix.png"
            )
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'model': model
        }
    
    def visualize_pca_decision_boundary(self, X_train, y_train, save_path=None):
        """
        Visualize decision boundary after PCA dimensionality reduction.
        
        Args:
            X_train: Training features
            y_train: Training labels
            save_path: Path to save the visualization
        """
        if self.optimized_model is None:
            raise ValueError("Optimized model not available. Call optimize() first.")
        
        # Dimensionality reduction to 2D
        pca = PCA(n_components=2)
        X_train_2d = pca.fit_transform(X_train)
        
        # Train model on 2D data
        model_2d = RandomForestClassifier(**self.best_params, random_state=self.random_state)
        model_2d.fit(X_train_2d, y_train)
        
        if save_path is None:
            save_path = OUTPUTS_DIR / "random_forest_pca_decision_boundary.png"
        
        plot_pca_decision_boundary(
            model_2d, 
            X_train_2d, 
            y_train, 
            pca, 
            NUMERIC_COLUMNS,
            save_path
        )

