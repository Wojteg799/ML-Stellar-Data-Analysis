"""
Visualization module.
Contains all plotting and visualization functions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from src.config import NUMERIC_COLUMNS, OUTPUTS_DIR

# Set style
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")


def plot_distributions(df, save_path=None):
    """
    Plot histograms of numeric variables.
    
    Args:
        df: DataFrame with numeric columns
        save_path: Path to save the figure. If None, saves to outputs directory.
    """
    numeric_df = df.select_dtypes(include=['number'])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx, col in enumerate(NUMERIC_COLUMNS):
        if col in numeric_df.columns:
            axes[idx].hist(numeric_df[col], bins=20, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
    
    plt.suptitle("Distribution of Values in the Dataset", fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path is None:
        save_path = OUTPUTS_DIR / "distributions.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Distribution plot saved to {save_path}")
    plt.close()


def plot_3d_scatter(df, save_path=None):
    """
    Create 3D scatter plot of stars.
    
    Args:
        df: DataFrame with star data
        save_path: Path to save the figure. If None, saves to outputs directory.
    """
    fig = px.scatter_3d(
        df, 
        x='Temperature (K)', 
        y='Luminosity(L/Lo)', 
        z='Radius(R/Ro)',
        color='Star type', 
        title='3D Visualization of Star Types',
        labels={'Star type': 'Star type'}
    )
    
    if save_path is None:
        save_path = OUTPUTS_DIR / "3d_scatter.html"
    
    fig.write_html(str(save_path))
    print(f"3D scatter plot saved to {save_path}")


def plot_correlation_matrix(df, save_path=None):
    """
    Plot correlation matrix heatmap.
    
    Args:
        df: DataFrame with numeric columns
        save_path: Path to save the figure. If None, saves to outputs directory.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[NUMERIC_COLUMNS].corr(), annot=True, cmap='coolwarm', fmt='.2f', 
                square=True, linewidths=0.5)
    plt.title("Correlation Matrix", fontsize=14, pad=20)
    plt.tight_layout()
    
    if save_path is None:
        save_path = OUTPUTS_DIR / "correlation_matrix.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Correlation matrix saved to {save_path}")
    plt.close()


def plot_hr_diagram(df, save_path=None):
    """
    Create Hertzsprung-Russell diagram.
    
    Args:
        df: DataFrame with star data
        save_path: Path to save the figure. If None, saves to outputs directory.
    """
    fig = px.scatter(
        df, 
        x='Temperature (K)', 
        y='Absolute magnitude(Mv)',
        color='Star type',
        title='H-R Diagram (Hertzsprung-Russell)',
        labels={
            'Temperature (K)': 'Temperature (K)', 
            'Absolute magnitude(Mv)': 'Absolute magnitude (Mv)'
        },
        hover_data=['Luminosity(L/Lo)', 'Radius(R/Ro)']
    )
    fig.update_xaxes(autorange="reversed")  # Temperature decreases to the right
    fig.update_yaxes(autorange="reversed")  # Magnitude is lower for brighter stars
    
    if save_path is None:
        save_path = OUTPUTS_DIR / "hr_diagram.html"
    
    fig.write_html(str(save_path))
    print(f"H-R diagram saved to {save_path}")


def plot_ternary(df, save_path=None):
    """
    Create ternary plot of star parameters.
    
    Args:
        df: DataFrame with star data
        save_path: Path to save the figure. If None, saves to outputs directory.
    """
    fig = px.scatter_ternary(
        df, 
        a='Luminosity(L/Lo)', 
        b='Radius(R/Ro)', 
        c='Absolute magnitude(Mv)',
        color='Star type', 
        title='Ternary Plot of Star Parameters'
    )
    
    if save_path is None:
        save_path = OUTPUTS_DIR / "ternary_plot.html"
    
    fig.write_html(str(save_path))
    print(f"Ternary plot saved to {save_path}")


def plot_decision_tree(tree, feature_names, save_path=None):
    """
    Visualize decision tree.
    
    Args:
        tree: Trained DecisionTreeClassifier
        feature_names: List of feature names
        save_path: Path to save the figure. If None, saves to outputs directory.
    """
    plt.figure(figsize=(20, 10))
    from sklearn.tree import plot_tree
    plot_tree(tree, filled=True, feature_names=feature_names, 
              class_names=[str(i) for i in range(6)], fontsize=8)
    plt.title("Decision Tree Visualization", fontsize=14)
    plt.tight_layout()
    
    if save_path is None:
        save_path = OUTPUTS_DIR / "decision_tree.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Decision tree visualization saved to {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, title, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        save_path: Path to save the figure. If None, saves to outputs directory.
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, 
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Predicted class', fontsize=12)
    plt.ylabel('True class', fontsize=12)
    plt.tight_layout()
    
    if save_path is None:
        # Create filename from title
        filename = title.lower().replace(' ', '_').replace('-', '_') + ".png"
        save_path = OUTPUTS_DIR / filename
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_pca_decision_boundary(model, X_train_2d, y_train, pca, feature_names, save_path=None):
    """
    Plot decision boundary after PCA dimensionality reduction.
    
    Args:
        model: Trained classifier
        X_train_2d: 2D training data after PCA
        y_train: Training labels
        pca: Fitted PCA object
        feature_names: List of feature names
        save_path: Path to save the figure. If None, saves to outputs directory.
    """
    from sklearn.decomposition import PCA
    
    explained_variance = pca.explained_variance_ratio_
    
    x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
    y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(15, 10))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    scatter = plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, 
                         alpha=0.8, cmap='viridis', edgecolors='black', linewidths=0.5)
    legend1 = plt.legend(*scatter.legend_elements(),
                        title="Star Types",
                        loc="upper right")
    plt.gca().add_artist(legend1)
    
    plt.xlabel(f'First Principal Component ({explained_variance[0]*100:.1f}% variance)', fontsize=12)
    plt.ylabel(f'Second Principal Component ({explained_variance[1]*100:.1f}% variance)', fontsize=12)
    plt.title('Random Forest Decision Boundary (after PCA dimensionality reduction)', fontsize=14)
    
    # Print feature importance on principal components
    feature_importance = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=feature_names
    )
    print("\nInfluence of features on Principal Components (PCA weights):")
    print(feature_importance)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = OUTPUTS_DIR / "pca_decision_boundary.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"PCA decision boundary plot saved to {save_path}")
    plt.close()


def plot_model_comparison(y_test, predictions_dict, save_path=None):
    """
    Plot comparison of confusion matrices for multiple models.
    
    Args:
        y_test: True test labels
        predictions_dict: Dictionary with model names as keys and predictions as values
        save_path: Path to save the figure. If None, saves to outputs directory.
    """
    from sklearn.metrics import confusion_matrix
    
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 6))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], 
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        axes[idx].set_title(model_name, fontsize=14, pad=15)
        axes[idx].set_xlabel('Predicted class', fontsize=12)
        axes[idx].set_ylabel('True class', fontsize=12)
    
    plt.suptitle('Comparison of Confusion Matrices for Optimized Models', 
                 fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path is None:
        save_path = OUTPUTS_DIR / "model_comparison_confusion_matrices.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Model comparison plot saved to {save_path}")
    plt.close()


def create_all_visualizations(df, save_to_outputs=True):
    """
    Create all visualizations and save them.
    
    Args:
        df: DataFrame with star data
        save_to_outputs: Whether to save plots to outputs directory
    """
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    
    if save_to_outputs:
        plot_distributions(df)
        plot_3d_scatter(df)
        plot_correlation_matrix(df)
        plot_hr_diagram(df)
        plot_ternary(df)
    else:
        # Just show plots without saving
        plot_distributions(df, save_path=False)
        plot_correlation_matrix(df, save_path=False)
        plt.show()

