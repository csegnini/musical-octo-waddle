"""
Advanced Feature Importance and Model Interpretability Module

This module provides comprehensive feature importance analysis and model interpretability tools.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List, Optional, Union, Tuple
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

class FeatureImportanceAnalyzer:
    """
    Comprehensive feature importance analysis for machine learning models.
    
    Provides multiple methods for analyzing feature importance including:
    - Built-in model feature importance
    - Permutation importance
    - SHAP values (if available)
    - Correlation analysis
    - Statistical significance testing
    """
    
    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        Initialize feature importance analyzer.
        
        Args:
            feature_names: Names of features for better visualization
        """
        self.feature_names = feature_names
        self.importance_results = {}
    
    def analyze_builtin_importance(self, model, method_name: str = "builtin") -> Optional[np.ndarray]:
        """
        Extract built-in feature importance from model.
        
        Args:
            model: Trained model with feature importance
            method_name: Name for this importance method
            
        Returns:
            Feature importance array or None
        """
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                self.importance_results[method_name] = importance
                return importance
            elif hasattr(model, 'coef_'):
                # For linear models, use absolute coefficients
                importance = np.abs(model.coef_).flatten()
                self.importance_results[method_name] = importance
                return importance
            else:
                print(f"⚠️ Model does not have built-in feature importance")
                return None
        except Exception as e:
            print(f"❌ Error extracting built-in importance: {e}")
            return None
    
    def analyze_permutation_importance(self, model, X: np.ndarray, y: np.ndarray,
                                     scoring: str = 'accuracy', n_repeats: int = 10,
                                     random_state: int = 42) -> Optional[np.ndarray]:
        """
        Calculate permutation importance.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target values
            scoring: Scoring metric
            n_repeats: Number of permutation repeats
            random_state: Random state for reproducibility
            
        Returns:
            Permutation importance array or None
        """
        try:
            print(f"🔄 Calculating permutation importance...")
            
            perm_importance = permutation_importance(
                model, X, y, scoring=scoring, n_repeats=n_repeats,
                random_state=random_state, n_jobs=-1
            )
            
            # Handle both Bunch and dict[str, Bunch] cases
            if isinstance(perm_importance, dict):
                # For multioutput estimators, select the first output
                first_key = list(perm_importance.keys())[0]
                importance = perm_importance[first_key].importances_mean
                self.importance_results['permutation'] = importance
                self.importance_results['permutation_std'] = perm_importance[first_key].importances_std
            else:
                importance = perm_importance.importances_mean
                self.importance_results['permutation'] = importance
                self.importance_results['permutation_std'] = perm_importance.importances_std
            
            print(f"✅ Permutation importance calculated")
            return importance
            
        except Exception as e:
            print(f"❌ Error calculating permutation importance: {e}")
            return None
    
    def analyze_correlation_importance(self, X: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculate feature importance based on correlation with target.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Correlation-based importance array or None
        """
        try:
            print(f"🔄 Calculating correlation-based importance...")
            
            # Calculate correlations
            correlations = []
            for i in range(X.shape[1]):
                corr = np.corrcoef(X[:, i], y)[0, 1]
                correlations.append(abs(corr))  # Use absolute correlation
            
            importance = np.array(correlations)
            self.importance_results['correlation'] = importance
            
            print(f"✅ Correlation importance calculated")
            return importance
            
        except Exception as e:
            print(f"❌ Error calculating correlation importance: {e}")
            return None
    
    def analyze_univariate_importance(self, X: np.ndarray, y: np.ndarray, 
                                    problem_type: str = 'classification') -> Optional[np.ndarray]:
        """
        Calculate univariate statistical importance.
        
        Args:
            X: Feature matrix
            y: Target values
            problem_type: 'classification' or 'regression'
            
        Returns:
            Univariate importance array or None
        """
        try:
            print(f"🔄 Calculating univariate importance...")
            
            if problem_type == 'classification':
                from sklearn.feature_selection import chi2, f_classif
                # Use F-statistic for classification
                f_scores, p_values = f_classif(X, y)
                importance = f_scores
            else:
                from sklearn.feature_selection import f_regression
                # Use F-statistic for regression
                f_scores, p_values = f_regression(X, y)
                importance = f_scores
            
            # Normalize importance scores
            importance = importance / np.max(importance)
            
            self.importance_results['univariate'] = importance
            self.importance_results['univariate_pvalues'] = p_values
            
            print(f"✅ Univariate importance calculated")
            return importance
            
        except Exception as e:
            print(f"❌ Error calculating univariate importance: {e}")
            return None
    
    def compare_importance_methods(self) -> pd.DataFrame:
        """
        Compare different importance methods in a DataFrame.
        
        Returns:
            DataFrame with importance scores from different methods
        """
        if not self.importance_results:
            print("⚠️ No importance results available. Run analysis methods first.")
            return pd.DataFrame()
        
        # Create comparison DataFrame
        comparison_data = {}
        
        # Get feature names
        if self.feature_names:
            comparison_data['Feature'] = self.feature_names
        else:
            n_features = len(list(self.importance_results.values())[0])
            comparison_data['Feature'] = [f'Feature_{i}' for i in range(n_features)]
        
        # Add importance scores from different methods
        for method, importance in self.importance_results.items():
            if not method.endswith('_std') and not method.endswith('_pvalues'):
                comparison_data[method.capitalize()] = importance
        
        df = pd.DataFrame(comparison_data)
        
        # Calculate average importance across methods
        importance_cols = [col for col in df.columns if col != 'Feature']
        if importance_cols:
            df['Average'] = df[importance_cols].mean(axis=1)
            df = df.sort_values('Average', ascending=False)
        
        return df
    
    def plot_feature_importance(self, methods: Optional[List[str]] = None, 
                              top_k: int = 15, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot feature importance from different methods.
        
        Args:
            methods: List of methods to plot (None for all)
            top_k: Number of top features to show
            figsize: Figure size
        """
        if not self.importance_results:
            print("⚠️ No importance results to plot. Run analysis methods first.")
            return
        
        # Filter methods
        if methods is None:
            methods = [m for m in self.importance_results.keys() 
                      if not m.endswith('_std') and not m.endswith('_pvalues')]
        
        available_methods = [m for m in methods if m in self.importance_results]
        
        if not available_methods:
            print("⚠️ None of the specified methods are available.")
            return
        
        # Create subplots
        n_methods = len(available_methods)
        fig, axes = plt.subplots(1, n_methods, figsize=figsize)
        
        if n_methods == 1:
            axes = [axes]
        
        for i, method in enumerate(available_methods):
            importance = self.importance_results[method]
            
            # Get top k features
            top_indices = np.argsort(importance)[::-1][:top_k]
            top_importance = importance[top_indices]
            
            # Get feature names
            if self.feature_names:
                top_features = [self.feature_names[j] for j in top_indices]
            else:
                top_features = [f'Feature_{j}' for j in top_indices]
            
            # Plot
            axes[i].barh(range(len(top_importance)), top_importance)
            axes[i].set_yticks(range(len(top_importance)))
            axes[i].set_yticklabels(top_features)
            axes[i].set_xlabel('Importance Score')
            axes[i].set_title(f'{method.capitalize()} Importance')
            axes[i].invert_yaxis()
            
            # Add value labels
            for j, (feature, score) in enumerate(zip(top_features, top_importance)):
                axes[i].text(score + 0.01 * max(top_importance), j, f'{score:.3f}', 
                           va='center', fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    def plot_importance_comparison(self, top_k: int = 15, 
                                 figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot comparison of feature importance across methods.
        
        Args:
            top_k: Number of top features to show
            figsize: Figure size
        """
        df = self.compare_importance_methods()
        
        if df.empty:
            return
        
        # Get top k features by average importance
        top_features = df.head(top_k)
        
        # Prepare data for plotting
        importance_cols = [col for col in df.columns if col not in ['Feature', 'Average']]
        
        if not importance_cols:
            print("⚠️ No importance methods to compare")
            return
        
        # Create grouped bar plot
        x = np.arange(len(top_features))
        width = 0.8 / len(importance_cols)
        
        plt.figure(figsize=figsize)
        
        for i, method in enumerate(importance_cols):
            plt.bar(x + i * width, top_features[method], width, 
                   label=method, alpha=0.8)
        
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.title(f'Feature Importance Comparison (Top {top_k} Features)')
        plt.xticks(x + width * (len(importance_cols) - 1) / 2, 
                  top_features['Feature'].tolist(), rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
    
    def plot_importance_heatmap(self, top_k: int = 20, 
                              figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot heatmap of feature importance across methods.
        
        Args:
            top_k: Number of top features to show
            figsize: Figure size
        """
        df = self.compare_importance_methods()
        
        if df.empty:
            return
        
        # Get top k features
        top_features = df.head(top_k)
        
        # Prepare data for heatmap
        importance_cols = [col for col in df.columns if col not in ['Feature', 'Average']]
        
        if not importance_cols:
            print("⚠️ No importance methods to compare")
            return
        
        # Create heatmap data
        heatmap_data = top_features[importance_cols].T
        heatmap_data.columns = top_features['Feature']
        
        # Normalize each row (method) to 0-1 scale for better comparison
        heatmap_data_norm = heatmap_data.div(heatmap_data.max(axis=1), axis=0)
        
        plt.figure(figsize=figsize)
        sns.heatmap(heatmap_data_norm, annot=True, fmt='.2f', cmap='YlOrRd',
                   cbar_kws={'label': 'Normalized Importance'})
        plt.title(f'Feature Importance Heatmap (Top {top_k} Features)')
        plt.xlabel('Features')
        plt.ylabel('Importance Methods')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    def generate_importance_report(self) -> str:
        """
        Generate a comprehensive feature importance report.
        
        Returns:
            Formatted text report
        """
        if not self.importance_results:
            return "No importance analysis results available."
        
        df = self.compare_importance_methods()
        
        report = "FEATURE IMPORTANCE ANALYSIS REPORT\n"
        report += "=" * 50 + "\n\n"
        
        # Summary statistics
        report += "Analysis Summary:\n"
        report += "-" * 20 + "\n"
        report += f"Number of features analyzed: {len(df)}\n"
        report += f"Number of importance methods: {len([m for m in self.importance_results.keys() if not m.endswith('_std') and not m.endswith('_pvalues')])}\n\n"
        
        # Top features
        if not df.empty:
            report += "Top 10 Most Important Features:\n"
            report += "-" * 35 + "\n"
            top_10 = df.head(10)
            
            for i, (_, row) in enumerate(top_10.iterrows(), 1):
                report += f"{i:2d}. {row['Feature']:<20} (Avg: {row.get('Average', 0):.4f})\n"
        
        # Method-specific insights
        report += f"\nMethod-Specific Analysis:\n"
        report += "-" * 25 + "\n"
        
        for method, importance in self.importance_results.items():
            if not method.endswith('_std') and not method.endswith('_pvalues'):
                report += f"\n{method.capitalize()} Importance:\n"
                top_idx = np.argsort(importance)[::-1][:5]
                
                for i, idx in enumerate(top_idx, 1):
                    feature_name = self.feature_names[idx] if self.feature_names else f'Feature_{idx}'
                    report += f"  {i}. {feature_name}: {importance[idx]:.4f}\n"
        
        return report
    
    def save_importance_results(self, filepath: str) -> None:
        """
        Save importance analysis results to CSV file.
        
        Args:
            filepath: Path to save the CSV file
        """
        df = self.compare_importance_methods()
        
        if not df.empty:
            df.to_csv(filepath, index=False)
            print(f"✅ Feature importance results saved to {filepath}")
        else:
            print("⚠️ No results to save")


def analyze_model_feature_importance(model, X: np.ndarray, y: np.ndarray,
                                   feature_names: Optional[List[str]] = None,
                                   problem_type: str = 'classification') -> FeatureImportanceAnalyzer:
    """
    Comprehensive feature importance analysis for a trained model.
    
    Args:
        model: Trained machine learning model
        X: Feature matrix
        y: Target values
        feature_names: Names of features
        problem_type: 'classification' or 'regression'
        
    Returns:
        FeatureImportanceAnalyzer with results
    """
    analyzer = FeatureImportanceAnalyzer(feature_names)
    
    print(f"🔍 Comprehensive Feature Importance Analysis")
    print("=" * 50)
    
    # Built-in importance
    print(f"\n1️⃣ Analyzing built-in feature importance...")
    analyzer.analyze_builtin_importance(model)
    
    # Permutation importance
    print(f"\n2️⃣ Analyzing permutation importance...")
    scoring = 'accuracy' if problem_type == 'classification' else 'r2'
    analyzer.analyze_permutation_importance(model, X, y, scoring=scoring)
    
    # Correlation importance
    print(f"\n3️⃣ Analyzing correlation-based importance...")
    analyzer.analyze_correlation_importance(X, y)
    
    # Univariate importance
    print(f"\n4️⃣ Analyzing univariate statistical importance...")
    analyzer.analyze_univariate_importance(X, y, problem_type)
    
    print(f"\n✅ Feature importance analysis completed!")
    
    return analyzer


# Export main components
__all__ = [
    'FeatureImportanceAnalyzer',
    'analyze_model_feature_importance'
]
