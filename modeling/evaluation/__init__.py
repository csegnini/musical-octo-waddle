"""
Model Evaluation Package

This package provides comprehensive evaluation capabilities for machine learning models
including metrics calculation, validation, visualization, and reporting.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple, Literal
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score, validation_curve, learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..base import BaseEvaluator, ModelType, ProblemType

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of evaluation metrics."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    RANKING = "ranking"


@dataclass
class EvaluationResult:
    """Result of model evaluation."""
    model_name: str
    metrics: Dict[str, float]
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[str] = None
    feature_importance: Optional[Dict[str, float]] = None
    cross_validation_scores: Optional[Dict[str, List[float]]] = None
    validation_curves: Optional[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None
    learning_curves: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None


class RegressionEvaluator(BaseEvaluator):
    """
    Evaluator for regression models.
    
    Provides comprehensive evaluation metrics and visualizations
    for regression problems.
    """
    
    def __init__(self, metrics: Optional[List[str]] = None):
        """
        Initialize regression evaluator.
        
        Args:
            metrics: List of metrics to calculate
        """
        default_metrics = ['mse', 'mae', 'r2', 'rmse']
        super().__init__(metrics or default_metrics)
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate regression predictions.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metric values
        """
        results = {}
        
        try:
            if 'mse' in self.metrics:
                results['mse'] = mean_squared_error(y_true, y_pred)
            
            if 'mae' in self.metrics:
                results['mae'] = mean_absolute_error(y_true, y_pred)
            
            if 'r2' in self.metrics:
                results['r2'] = r2_score(y_true, y_pred)
            
            if 'rmse' in self.metrics:
                results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            
            # Additional metrics
            if 'mape' in self.metrics:
                # Mean Absolute Percentage Error
                mask = y_true != 0
                if np.any(mask):
                    results['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            
            if 'max_error' in self.metrics:
                results['max_error'] = np.max(np.abs(y_true - y_pred))
            
            if 'explained_variance' in self.metrics:
                from sklearn.metrics import explained_variance_score
                results['explained_variance'] = explained_variance_score(y_true, y_pred)
            
        except Exception as e:
            logger.error(f"Error calculating regression metrics: {e}")
        
        return results
    
    def generate_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Formatted evaluation report
        """
        metrics = self.evaluate(y_true, y_pred)
        
        report = "REGRESSION EVALUATION REPORT\n"
        report += "=" * 40 + "\n\n"
        
        report += "Metrics:\n"
        report += "-" * 20 + "\n"
        for metric, value in metrics.items():
            report += f"{metric.upper()}: {value:.6f}\n"
        
        # Additional statistics
        residuals = y_true - y_pred
        report += f"\nResidual Statistics:\n"
        report += "-" * 20 + "\n"
        report += f"Mean Residual: {np.mean(residuals):.6f}\n"
        report += f"Std Residual: {np.std(residuals):.6f}\n"
        report += f"Min Residual: {np.min(residuals):.6f}\n"
        report += f"Max Residual: {np.max(residuals):.6f}\n"
        
        return report
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None) -> None:
        """
        Plot predictions vs actual values.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Predictions vs Actual
        plt.subplot(2, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Predictions vs Actual')
        plt.grid(True, alpha=0.3)
        
        # Residual plot
        plt.subplot(2, 2, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)
        
        # Residual distribution
        plt.subplot(2, 2, 3)
        plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residual Distribution')
        plt.grid(True, alpha=0.3)
        
        # Q-Q plot
        plt.subplot(2, 2, 4)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot (Normal)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Regression plots saved to {save_path}")
        
        plt.show()


class ClassificationEvaluator(BaseEvaluator):
    """
    Enhanced evaluator for classification models.
    
    Provides comprehensive evaluation metrics, ROC curves, precision-recall curves,
    confusion matrices, and detailed reporting for binary and multiclass classification.
    """
    
    from typing import Literal

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        average: Literal['micro', 'macro', 'samples', 'weighted', 'binary'] = 'weighted'
    ):
        """
        Initialize classification evaluator.
        
        Args:
            metrics: List of metrics to calculate
            average: Averaging strategy for multi-class classification
        """
        default_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        super().__init__(metrics or default_metrics)
        self.average: Literal['micro', 'macro', 'samples', 'weighted', 'binary'] = average
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate classification predictions with comprehensive metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted class labels
            y_proba: Predicted class probabilities (optional)
            
        Returns:
            Dictionary of metric values
        """
        results = {}
        
        try:
            # Basic classification metrics
            if 'accuracy' in self.metrics:
                results['accuracy'] = accuracy_score(y_true, y_pred)
            
            if 'precision' in self.metrics:
                results['precision'] = precision_score(y_true, y_pred, average=self.average, zero_division=0)
            
            if 'recall' in self.metrics:
                results['recall'] = recall_score(y_true, y_pred, average=self.average, zero_division=0)
            
            if 'f1' in self.metrics:
                results['f1'] = f1_score(y_true, y_pred, average=self.average, zero_division=0)
            
            # ROC AUC and probability-based metrics
            if y_proba is not None:
                try:
                    n_classes = len(np.unique(y_true))
                    
                    if 'roc_auc' in self.metrics:
                        if n_classes == 2:  # Binary classification
                            results['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                        else:  # Multiclass classification
                            # Only pass valid average values for multiclass roc_auc_score
                            from typing import cast
                            if self.average in ['micro', 'macro', 'samples', 'weighted']:
                                roc_auc_average = cast(Literal['micro', 'macro', 'samples', 'weighted'], self.average)
                            else:
                                roc_auc_average = 'weighted'
                            results['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average=roc_auc_average)
                    
                    if 'log_loss' in self.metrics:
                        from sklearn.metrics import log_loss
                        results['log_loss'] = log_loss(y_true, y_proba)
                        
                except (ValueError, IndexError) as e:
                    logger.warning(f"Could not compute probability-based metrics: {e}")
            
            # Multi-class specific metrics
            n_classes = len(np.unique(y_true))
            if n_classes > 2:
                if 'precision_macro' in self.metrics:
                    results['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
                if 'recall_macro' in self.metrics:
                    results['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
                if 'f1_macro' in self.metrics:
                    results['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
                if 'precision_micro' in self.metrics:
                    results['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
                if 'recall_micro' in self.metrics:
                    results['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
                if 'f1_micro' in self.metrics:
                    results['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
                    
        except Exception as e:
            logger.error(f"Error in classification evaluation: {e}")
            
        return results
    
    def generate_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       target_names: Optional[List[str]] = None,
                       model_name: str = "Classification Model") -> str:
        """
        Generate comprehensive classification report.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            target_names: Names of target classes
            model_name: Name of the model
            
        Returns:
            Formatted evaluation report
        """
        try:
            metrics = self.evaluate(y_true, y_pred)
            
            report = f"\n=== {model_name} Classification Report ===\n"
            report += f"Total Samples: {len(y_true)}\n"
            report += f"Number of Classes: {len(np.unique(y_true))}\n\n"
            
            # Overall metrics
            report += "Overall Metrics:\n"
            for metric, value in metrics.items():
                report += f"  {metric.upper().replace('_', ' ')}: {value:.4f}\n"
            
            # Detailed sklearn classification report
            report += f"\nDetailed Classification Report:\n"
            sklearn_report = classification_report(
                y_true, y_pred,
                target_names=target_names,
                zero_division=0
            )
            # Ensure sklearn_report is a string
            if isinstance(sklearn_report, str):
                report += sklearn_report
            else:
                report += str(sklearn_report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating classification report: {e}")
            return f"Error generating report: {e}"
    
    def get_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Get confusion matrix."""
        return confusion_matrix(y_true, y_pred)
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: Optional[List[str]] = None,
                             title: str = "Confusion Matrix",
                             normalize: bool = False) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            class_names: Names of classes
            title: Plot title
            normalize: Whether to normalize the matrix
        """
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                
            plt.figure(figsize=(8, 6))
            labels = class_names if class_names is not None else [str(x) for x in np.unique(y_true)]
            sns.heatmap(cm, 
                       annot=True, 
                       fmt='.2f' if normalize else 'd', 
                       cmap='Blues',
                       xticklabels=labels,
                       yticklabels=labels)
            plt.title(title)
            plt.xlabel('Predicted Class')
            plt.ylabel('True Class')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")
    
    def plot_roc_curves(self, y_true: np.ndarray, y_proba: np.ndarray,
                       class_names: Optional[List[str]] = None,
                       title: str = "ROC Curves") -> Dict[str, Tuple[np.ndarray, np.ndarray, float]]:
        """
        Plot ROC curves for binary or multiclass classification.
        
        Args:
            y_true: True target values
            y_proba: Predicted class probabilities
            class_names: Names of classes
            title: Plot title
            
        Returns:
            Dictionary containing ROC curve data
        """
        try:
            classes = np.unique(y_true)
            n_classes = len(classes)
            
            plt.figure(figsize=(10, 8))
            roc_data = {}
            
            if n_classes == 2:
                # Binary classification
                fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
                auc = roc_auc_score(y_true, y_proba[:, 1])
                
                plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
                roc_data['binary'] = (fpr, tpr, auc)
                
            else:
                # Multiclass classification
                from sklearn.preprocessing import label_binarize
                
                y_true_bin = label_binarize(y_true, classes=classes)
                # Convert to dense numpy array - use a type-agnostic approach
                try:
                    # For sparse matrices, use the array attribute or conversion method
                    if hasattr(y_true_bin, 'toarray'):
                        y_true_bin = getattr(y_true_bin, 'toarray')()
                    else:
                        y_true_bin = np.array(y_true_bin)
                except Exception:
                    y_true_bin = np.asarray(y_true_bin)
                
                for i, class_label in enumerate(classes):
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                    auc = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
                    
                    class_name = class_names[i] if class_names else f'Class {class_label}'
                    plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})', linewidth=2)
                    roc_data[f'class_{class_label}'] = (fpr, tpr, auc)
            
            plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(title)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            return roc_data
            
        except Exception as e:
            logger.error(f"Error plotting ROC curves: {e}")
            return {}
    
    def plot_precision_recall_curves(self, y_true: np.ndarray, y_proba: np.ndarray,
                                    class_names: Optional[List[str]] = None,
                                    title: str = "Precision-Recall Curves") -> Dict[str, Tuple[np.ndarray, np.ndarray, float]]:
        """
        Plot precision-recall curves.
        
        Args:
            y_true: True target values
            y_proba: Predicted class probabilities
            class_names: Names of classes
            title: Plot title
            
        Returns:
            Dictionary containing PR curve data
        """
        try:
            classes = np.unique(y_true)
            n_classes = len(classes)
            
            plt.figure(figsize=(10, 8))
            pr_data = {}
            
            if n_classes == 2:
                # Binary classification
                precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
                from sklearn.metrics import auc
                ap = auc(recall, precision)  # More stable than trapz
                
                plt.plot(recall, precision, label=f'PR Curve (AP = {ap:.3f})', linewidth=2)
                pr_data['binary'] = (precision, recall, ap)
                
            else:
                # Multiclass classification
                from sklearn.preprocessing import label_binarize
                y_true_bin = label_binarize(y_true, classes=classes)
                # Convert to dense numpy array - use a type-agnostic approach
                try:
                    # For sparse matrices, use the array attribute or conversion method
                    if hasattr(y_true_bin, 'toarray'):
                        y_true_bin = getattr(y_true_bin, 'toarray')()
                    else:
                        y_true_bin = np.array(y_true_bin)
                except Exception:
                    y_true_bin = np.asarray(y_true_bin)
                
                for i, class_label in enumerate(classes):
                    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
                    from sklearn.metrics import auc
                    ap = auc(recall, precision)  # More stable than trapz
                    
                    class_name = class_names[i] if class_names else f'Class {class_label}'
                    plt.plot(recall, precision, label=f'{class_name} (AP = {ap:.3f})', linewidth=2)
                    pr_data[f'class_{class_label}'] = (precision, recall, ap)
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(title)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            return pr_data
            
        except Exception as e:
            logger.error(f"Error plotting precision-recall curves: {e}")
            return {}


class ModelValidator:
    """
    Comprehensive model validation utility.

    Provides cross-validation, validation curves, learning curves,
    and other validation techniques.
    """

    def __init__(self, cv: int = 5, random_state: int = 42):
        """
        Initialize model validator.

        Args:
            cv: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        self.cv = cv
        self.random_state = random_state

    def cross_validate(self, model, X: np.ndarray, y: np.ndarray,
                      scoring: Union[str, List[str]] = 'accuracy') -> Dict[str, List[float]]:
        """
        Perform cross-validation.

        Args:
            model: Model to validate
            X: Features
            y: Targets
            scoring: Scoring metric(s)

        Returns:
            Cross-validation scores
        """
        if isinstance(scoring, str):
            scoring = [scoring]

        results = {}

        for score in scoring:
            try:
                scores = cross_val_score(model, X, y, cv=self.cv, scoring=score)
                results[score] = scores.tolist()
                logger.info(f"{score.upper()} CV scores: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            except Exception as e:
                logger.error(f"Error in cross-validation for {score}: {e}")

        return results

    def validation_curve_analysis(self, model, X: np.ndarray, y: np.ndarray,
                                 param_name: str, param_range: List[Any],
                                 scoring: str = 'accuracy') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate validation curve.

        Args:
            model: Model to validate
            X: Features
            y: Targets
            param_name: Parameter to vary
            param_range: Range of parameter values
            scoring: Scoring metric

        Returns:
            Tuple of (param_range, train_scores, validation_scores)
        """
        try:
            train_scores, val_scores = validation_curve(
                model, X, y, param_name=param_name, param_range=param_range,
                cv=self.cv, scoring=scoring, n_jobs=-1
            )

            return np.array(param_range), train_scores, val_scores

        except Exception as e:
            logger.error(f"Error generating validation curve: {e}")
            return np.array([]), np.array([]), np.array([])

    def learning_curve_analysis(self, model, X: np.ndarray, y: np.ndarray,
                               train_sizes: Optional[np.ndarray] = None,
                               scoring: str = 'accuracy') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate learning curve.

        Args:
            model: Model to validate
            X: Features
            y: Targets
            train_sizes: Training set sizes to use
            scoring: Scoring metric

        Returns:
            Tuple of (train_sizes, train_scores, validation_scores)
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        try:
            train_sizes_abs, train_scores, val_scores, _, _ = learning_curve(
                model, X, y, train_sizes=train_sizes, cv=self.cv,
                scoring=scoring, n_jobs=-1
            )

            return train_sizes_abs, train_scores, val_scores

        except Exception as e:
            logger.error(f"Error generating learning curve: {e}")
            return np.array([]), np.array([]), np.array([])

    def plot_validation_curve(self, param_range: np.ndarray, train_scores: np.ndarray,
                             val_scores: np.ndarray, param_name: str,
                             save_path: Optional[str] = None) -> None:
        """Plot validation curve."""
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(param_range, train_mean, 'o-', color='blue', label='Training score')
        plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')

        plt.plot(param_range, val_mean, 'o-', color='red', label='Cross-validation score')
        plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')

        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.title(f'Validation Curve ({param_name})')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Validation curve saved to {save_path}")

        plt.show()

    def plot_learning_curve(self, train_sizes: np.ndarray, train_scores: np.ndarray,
                           val_scores: np.ndarray, save_path: Optional[str] = None) -> None:
        """Plot learning curve."""
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')

        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Cross-validation score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')

        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title('Learning Curve')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Learning curve saved to {save_path}")

        plt.show()


class EvaluationSuite:
    """
    Comprehensive evaluation suite for any model type.
    """
    
    def __init__(self):
        """Initialize evaluation suite."""
        self.regression_evaluator = RegressionEvaluator()
        self.classification_evaluator = ClassificationEvaluator()
        self.validator = ModelValidator()
    
    def evaluate_model(self, model, X: np.ndarray, y: np.ndarray, 
                      model_type: ModelType, test_size: float = 0.2) -> EvaluationResult:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Targets
            model_type: Type of model
            test_size: Test set size
            
        Returns:
            Comprehensive evaluation result
        """
        from sklearn.model_selection import train_test_split
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train model if not already fitted
        if not getattr(model, 'is_fitted', False):
            model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate based on model type
        if model_type == ModelType.REGRESSION:
            metrics = self.regression_evaluator.evaluate(y_test, y_pred)
            report = self.regression_evaluator.generate_report(y_test, y_pred)
        else:
            metrics = self.classification_evaluator.evaluate(y_test, y_pred)
            report = self.classification_evaluator.generate_report(y_test, y_pred)
        
        # Cross-validation
        cv_scores = self.validator.cross_validate(model, X_train, y_train)
        
        # Feature importance
        feature_importance = None
        if hasattr(model, 'get_feature_importance'):
            feature_importance = model.get_feature_importance()
        
        return EvaluationResult(
            model_name=getattr(getattr(model, 'metadata', None), 'name', 'Unknown Model'),
            metrics=metrics,
            classification_report=report if model_type != ModelType.REGRESSION else None,
            feature_importance=feature_importance,
            cross_validation_scores=cv_scores
        )
