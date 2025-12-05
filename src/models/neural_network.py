"""
Neural Network model for star type classification.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

from src.config import NN_PARAMS, RANDOM_STATE, CV_FOLDS, SCORING, NN_MAX_ITER, OUTPUTS_DIR
from src.visualization import plot_confusion_matrix


class NeuralNetworkModel:
    """Neural Network classifier with optimization capabilities."""
    
    def __init__(self, random_state=None, max_iter=None):
        """
        Initialize Neural Network model.
        
        Args:
            random_state: Random state for reproducibility
            max_iter: Maximum number of iterations
        """
        if random_state is None:
            random_state = RANDOM_STATE
        if max_iter is None:
            max_iter = NN_MAX_ITER
        
        self.random_state = random_state
        self.max_iter = max_iter
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
        print("NEURAL NETWORK - BASELINE MODEL")
        print("="*50)
        
        self.model = MLPClassifier(max_iter=self.max_iter, random_state=self.random_state)
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
            param_grid = NN_PARAMS
        if cv is None:
            cv = CV_FOLDS
        if scoring is None:
            scoring = SCORING
        
        print("\n" + "="*50)
        print("NEURAL NETWORK - HYPERPARAMETER OPTIMIZATION")
        print("="*50)
        
        grid_search = GridSearchCV(
            MLPClassifier(max_iter=self.max_iter, random_state=self.random_state),
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
        
        print(f"\n{model_name} Neural Network Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        
        if save_plots and use_optimized:
            # Save confusion matrix
            plot_confusion_matrix(
                y_test, 
                predictions, 
                f"Neural Network - {model_name}",
                OUTPUTS_DIR / "neural_network_confusion_matrix.png"
            )
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'model': model
        }

