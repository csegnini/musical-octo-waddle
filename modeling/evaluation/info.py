"""
Model Evaluation Module Information

This comprehensive documentation covers the model evaluation module, which provides
extensive evaluation capabilities for machine learning models including metrics
calculation, validation, visualization, and detailed reporting.

The evaluation module is designed to support regression, classification, and other
ML model types with standardized interfaces and comprehensive evaluation suites.
"""

import inspect
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Import evaluation module components
try:
    from . import (
        MetricType, EvaluationResult, RegressionEvaluator, 
        ClassificationEvaluator, ModelValidator, EvaluationSuite
    )
    from ..base import BaseEvaluator, ModelType, ProblemType
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import evaluation components: {e}")
    IMPORTS_AVAILABLE = False


@dataclass
class EvaluationModuleInfo:
    """Information structure for the evaluation module."""
    name: str = "Model Evaluation Module"
    version: str = "2.0.0"
    description: str = "Comprehensive model evaluation and validation framework"
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    components: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)


class EvaluationModuleAnalyzer:
    """
    Analyzer for the Model Evaluation Module.
    
    This class provides comprehensive analysis and documentation of the evaluation
    module's components, capabilities, and usage patterns.
    """
    
    def __init__(self):
        """Initialize the evaluation module analyzer."""
        self.module_info = EvaluationModuleInfo()
        self._initialize_module_info()
    
    def _initialize_module_info(self):
        """Initialize comprehensive module information."""
        self.module_info.dependencies = [
            "numpy", "pandas", "scikit-learn", "matplotlib", "seaborn",
            "scipy", "dataclasses", "enum", "logging", "typing"
        ]
        
        self.module_info.capabilities = [
            "Regression Model Evaluation",
            "Classification Model Evaluation", 
            "Cross-Validation Analysis",
            "Validation Curves",
            "Learning Curves",
            "ROC Curve Analysis",
            "Precision-Recall Curves",
            "Confusion Matrix Generation",
            "Feature Importance Analysis",
            "Model Performance Visualization",
            "Comprehensive Reporting",
            "Multi-Class Support",
            "Probability-Based Metrics",
            "Model Comparison",
            "Statistical Validation"
        ]
        
        if IMPORTS_AVAILABLE:
            self._analyze_components()
    
    def _analyze_components(self):
        """Analyze all components in the evaluation module."""
        self.module_info.components = {
            "enums": self._analyze_enums(),
            "data_classes": self._analyze_data_classes(),
            "evaluators": self._analyze_evaluators(),
            "validators": self._analyze_validators(),
            "suites": self._analyze_suites()
        }
    
    def _analyze_enums(self) -> Dict[str, Any]:
        """Analyze enum classes in the module."""
        return {
            "MetricType": {
                "description": "Enumeration of different types of evaluation metrics",
                "values": [
                    "REGRESSION - Metrics for regression models",
                    "CLASSIFICATION - Metrics for classification models", 
                    "CLUSTERING - Metrics for clustering models",
                    "RANKING - Metrics for ranking models"
                ],
                "purpose": "Categorizes metrics by problem type for organized evaluation"
            }
        }
    
    def _analyze_data_classes(self) -> Dict[str, Any]:
        """Analyze data classes in the module."""
        return {
            "EvaluationResult": {
                "description": "Comprehensive container for model evaluation results",
                "fields": [
                    "model_name: str - Name of the evaluated model",
                    "metrics: Dict[str, float] - Calculated performance metrics",
                    "confusion_matrix: Optional[np.ndarray] - Confusion matrix for classification",
                    "classification_report: Optional[str] - Detailed classification report",
                    "feature_importance: Optional[Dict[str, float]] - Feature importance scores",
                    "cross_validation_scores: Optional[Dict[str, List[float]]] - CV scores",
                    "validation_curves: Optional[Dict] - Validation curve data",
                    "learning_curves: Optional[Tuple] - Learning curve data"
                ],
                "purpose": "Standardizes evaluation output format across all model types",
                "methods": []
            }
        }
    
    def _analyze_evaluators(self) -> Dict[str, Any]:
        """Analyze evaluator classes in the module."""
        evaluators = {}
        
        # Regression Evaluator
        evaluators["RegressionEvaluator"] = {
            "description": "Comprehensive evaluator for regression models",
            "inheritance": "BaseEvaluator",
            "purpose": "Provides specialized evaluation metrics and visualizations for regression problems",
            "key_features": [
                "Multiple regression metrics (MSE, MAE, R², RMSE, MAPE)",
                "Residual analysis and visualization",
                "Prediction vs actual plots",
                "Statistical validation",
                "Comprehensive reporting"
            ],
            "methods": {
                "__init__": {
                    "description": "Initialize regression evaluator with configurable metrics",
                    "parameters": ["metrics: Optional[List[str]] - Metrics to calculate"],
                    "default_metrics": ["mse", "mae", "r2", "rmse"]
                },
                "evaluate": {
                    "description": "Calculate regression metrics for predictions",
                    "parameters": [
                        "y_true: np.ndarray - True target values",
                        "y_pred: np.ndarray - Predicted values"
                    ],
                    "returns": "Dict[str, float] - Dictionary of metric values",
                    "supported_metrics": [
                        "mse - Mean Squared Error",
                        "mae - Mean Absolute Error", 
                        "r2 - R-squared Score",
                        "rmse - Root Mean Squared Error",
                        "mape - Mean Absolute Percentage Error",
                        "max_error - Maximum Error",
                        "explained_variance - Explained Variance Score"
                    ]
                },
                "generate_report": {
                    "description": "Generate comprehensive evaluation report",
                    "parameters": [
                        "y_true: np.ndarray - True target values",
                        "y_pred: np.ndarray - Predicted values"
                    ],
                    "returns": "str - Formatted evaluation report",
                    "includes": [
                        "All calculated metrics",
                        "Residual statistics",
                        "Performance summary"
                    ]
                },
                "plot_predictions": {
                    "description": "Create comprehensive prediction visualization plots",
                    "parameters": [
                        "y_true: np.ndarray - True target values",
                        "y_pred: np.ndarray - Predicted values",
                        "save_path: Optional[str] - Path to save plots"
                    ],
                    "visualizations": [
                        "Predictions vs Actual scatter plot",
                        "Residual plot",
                        "Residual distribution histogram",
                        "Q-Q plot for normality testing"
                    ]
                }
            },
            "usage_example": '''
# Initialize regression evaluator
evaluator = RegressionEvaluator(['mse', 'mae', 'r2'])

# Evaluate predictions
metrics = evaluator.evaluate(y_true, y_pred)

# Generate comprehensive report
report = evaluator.generate_report(y_true, y_pred)

# Create visualization plots
evaluator.plot_predictions(y_true, y_pred, 'regression_plots.png')
'''
        }
        
        # Classification Evaluator
        evaluators["ClassificationEvaluator"] = {
            "description": "Enhanced evaluator for classification models with comprehensive metrics",
            "inheritance": "BaseEvaluator",
            "purpose": "Provides specialized evaluation for binary and multiclass classification",
            "key_features": [
                "Standard classification metrics (accuracy, precision, recall, F1)",
                "ROC curve analysis with AUC calculation",
                "Precision-Recall curves",
                "Confusion matrix generation and visualization",
                "Multi-class and multi-label support",
                "Probability-based metrics",
                "Class-wise performance analysis"
            ],
            "methods": {
                "__init__": {
                    "description": "Initialize classification evaluator with configurable settings",
                    "parameters": [
                        "metrics: Optional[List[str]] - Metrics to calculate",
                        "average: Literal['micro', 'macro', 'samples', 'weighted', 'binary'] - Averaging strategy"
                    ],
                    "default_metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
                    "averaging_strategies": {
                        "micro": "Global average across all samples",
                        "macro": "Unweighted average across classes",
                        "weighted": "Weighted average by class support",
                        "binary": "For binary classification only",
                        "samples": "Average across samples (multilabel)"
                    }
                },
                "evaluate": {
                    "description": "Calculate comprehensive classification metrics",
                    "parameters": [
                        "y_true: np.ndarray - True target values",
                        "y_pred: np.ndarray - Predicted class labels",
                        "y_proba: Optional[np.ndarray] - Predicted class probabilities"
                    ],
                    "returns": "Dict[str, float] - Dictionary of metric values",
                    "supported_metrics": [
                        "accuracy - Overall accuracy",
                        "precision - Precision score",
                        "recall - Recall score", 
                        "f1 - F1 score",
                        "roc_auc - ROC AUC score",
                        "log_loss - Logarithmic loss",
                        "precision_macro/micro - Class-averaged precision",
                        "recall_macro/micro - Class-averaged recall",
                        "f1_macro/micro - Class-averaged F1"
                    ]
                },
                "generate_report": {
                    "description": "Generate detailed classification report",
                    "parameters": [
                        "y_true: np.ndarray - True target values",
                        "y_pred: np.ndarray - Predicted values",
                        "target_names: Optional[List[str]] - Class names",
                        "model_name: str - Model identifier"
                    ],
                    "returns": "str - Comprehensive formatted report",
                    "includes": [
                        "Overall performance metrics",
                        "Class-wise performance breakdown",
                        "Sklearn classification report",
                        "Sample and class counts"
                    ]
                },
                "get_confusion_matrix": {
                    "description": "Generate confusion matrix",
                    "parameters": [
                        "y_true: np.ndarray - True target values",
                        "y_pred: np.ndarray - Predicted values"
                    ],
                    "returns": "np.ndarray - Confusion matrix"
                },
                "plot_confusion_matrix": {
                    "description": "Visualize confusion matrix with heatmap",
                    "parameters": [
                        "y_true: np.ndarray - True target values",
                        "y_pred: np.ndarray - Predicted values",
                        "class_names: Optional[List[str]] - Class labels",
                        "title: str - Plot title",
                        "normalize: bool - Whether to normalize values"
                    ],
                    "features": [
                        "Seaborn heatmap visualization",
                        "Customizable class labels",
                        "Optional normalization",
                        "Professional styling"
                    ]
                },
                "plot_roc_curves": {
                    "description": "Plot ROC curves for binary or multiclass classification",
                    "parameters": [
                        "y_true: np.ndarray - True target values",
                        "y_proba: np.ndarray - Predicted probabilities",
                        "class_names: Optional[List[str]] - Class labels",
                        "title: str - Plot title"
                    ],
                    "returns": "Dict[str, Tuple] - ROC curve data for each class",
                    "features": [
                        "Automatic binary/multiclass handling",
                        "AUC calculation per class",
                        "Professional visualization",
                        "Random classifier baseline"
                    ]
                },
                "plot_precision_recall_curves": {
                    "description": "Plot precision-recall curves",
                    "parameters": [
                        "y_true: np.ndarray - True target values", 
                        "y_proba: np.ndarray - Predicted probabilities",
                        "class_names: Optional[List[str]] - Class labels",
                        "title: str - Plot title"
                    ],
                    "returns": "Dict[str, Tuple] - PR curve data for each class",
                    "features": [
                        "Average Precision calculation",
                        "Multi-class support",
                        "Area under curve computation",
                        "Clear visualization"
                    ]
                }
            },
            "usage_example": '''
# Initialize classification evaluator
evaluator = ClassificationEvaluator(['accuracy', 'f1', 'roc_auc'], average='weighted')

# Evaluate with probabilities
metrics = evaluator.evaluate(y_true, y_pred, y_proba)

# Generate comprehensive report
report = evaluator.generate_report(y_true, y_pred, class_names, "My Model")

# Visualize results
evaluator.plot_confusion_matrix(y_true, y_pred, class_names)
roc_data = evaluator.plot_roc_curves(y_true, y_proba, class_names)
pr_data = evaluator.plot_precision_recall_curves(y_true, y_proba, class_names)
'''
        }
        
        return evaluators
    
    def _analyze_validators(self) -> Dict[str, Any]:
        """Analyze validator classes in the module."""
        return {
            "ModelValidator": {
                "description": "Comprehensive model validation utility for cross-validation and curve analysis",
                "purpose": "Provides advanced validation techniques beyond basic metric calculation",
                "key_features": [
                    "K-fold cross-validation",
                    "Validation curve generation",
                    "Learning curve analysis", 
                    "Hyperparameter validation",
                    "Training progress visualization",
                    "Model generalization assessment"
                ],
                "methods": {
                    "__init__": {
                        "description": "Initialize validator with cross-validation settings",
                        "parameters": [
                            "cv: int = 5 - Number of cross-validation folds",
                            "random_state: int = 42 - Random state for reproducibility"
                        ]
                    },
                    "cross_validate": {
                        "description": "Perform comprehensive cross-validation",
                        "parameters": [
                            "model - Model to validate",
                            "X: np.ndarray - Features",
                            "y: np.ndarray - Targets", 
                            "scoring: Union[str, List[str]] - Scoring metric(s)"
                        ],
                        "returns": "Dict[str, List[float]] - CV scores for each metric",
                        "features": [
                            "Multiple scoring metrics support",
                            "Statistical summary (mean, std)",
                            "Detailed logging",
                            "Error handling"
                        ]
                    },
                    "validation_curve_analysis": {
                        "description": "Generate validation curves for hyperparameter tuning",
                        "parameters": [
                            "model - Model to validate",
                            "X: np.ndarray - Features",
                            "y: np.ndarray - Targets",
                            "param_name: str - Parameter to vary",
                            "param_range: List[Any] - Parameter values to test",
                            "scoring: str - Scoring metric"
                        ],
                        "returns": "Tuple[np.ndarray, np.ndarray, np.ndarray] - (param_range, train_scores, val_scores)",
                        "purpose": "Analyze model performance across parameter ranges to detect overfitting/underfitting"
                    },
                    "learning_curve_analysis": {
                        "description": "Generate learning curves to analyze training data requirements",
                        "parameters": [
                            "model - Model to validate",
                            "X: np.ndarray - Features", 
                            "y: np.ndarray - Targets",
                            "train_sizes: Optional[np.ndarray] - Training set sizes",
                            "scoring: str - Scoring metric"
                        ],
                        "returns": "Tuple[np.ndarray, np.ndarray, np.ndarray] - (train_sizes, train_scores, val_scores)",
                        "purpose": "Determine optimal training set size and detect overfitting"
                    },
                    "plot_validation_curve": {
                        "description": "Visualize validation curve with confidence intervals",
                        "parameters": [
                            "param_range: np.ndarray - Parameter values",
                            "train_scores: np.ndarray - Training scores",
                            "val_scores: np.ndarray - Validation scores", 
                            "param_name: str - Parameter name for labeling",
                            "save_path: Optional[str] - Path to save plot"
                        ],
                        "features": [
                            "Mean score lines",
                            "Standard deviation bands",
                            "Professional styling",
                            "Optional plot saving"
                        ]
                    },
                    "plot_learning_curve": {
                        "description": "Visualize learning curve with training/validation performance",
                        "parameters": [
                            "train_sizes: np.ndarray - Training set sizes",
                            "train_scores: np.ndarray - Training scores",
                            "val_scores: np.ndarray - Validation scores",
                            "save_path: Optional[str] - Path to save plot"
                        ],
                        "features": [
                            "Training vs validation comparison",
                            "Confidence intervals",
                            "Overfitting detection",
                            "Clear visualization"
                        ]
                    }
                },
                "usage_example": '''
# Initialize validator
validator = ModelValidator(cv=5, random_state=42)

# Perform cross-validation
cv_scores = validator.cross_validate(model, X, y, ['accuracy', 'f1'])

# Generate validation curve
param_range = [0.1, 1, 10, 100]
param_range, train_scores, val_scores = validator.validation_curve_analysis(
    model, X, y, 'C', param_range
)
validator.plot_validation_curve(param_range, train_scores, val_scores, 'C')

# Generate learning curve
train_sizes, train_scores, val_scores = validator.learning_curve_analysis(model, X, y)
validator.plot_learning_curve(train_sizes, train_scores, val_scores)
'''
            }
        }
    
    def _analyze_suites(self) -> Dict[str, Any]:
        """Analyze evaluation suite classes."""
        return {
            "EvaluationSuite": {
                "description": "Comprehensive evaluation suite integrating all evaluation components",
                "purpose": "Provides unified interface for complete model evaluation workflow",
                "key_features": [
                    "Automatic model type detection",
                    "Integrated evaluation pipeline",
                    "Cross-validation integration",
                    "Feature importance analysis",
                    "Comprehensive result packaging",
                    "Standardized evaluation workflow"
                ],
                "components": [
                    "RegressionEvaluator instance",
                    "ClassificationEvaluator instance", 
                    "ModelValidator instance"
                ],
                "methods": {
                    "__init__": {
                        "description": "Initialize evaluation suite with all evaluators",
                        "components_initialized": [
                            "regression_evaluator: RegressionEvaluator",
                            "classification_evaluator: ClassificationEvaluator",
                            "validator: ModelValidator"
                        ]
                    },
                    "evaluate_model": {
                        "description": "Perform comprehensive model evaluation with automatic type detection",
                        "parameters": [
                            "model - Model to evaluate",
                            "X: np.ndarray - Features",
                            "y: np.ndarray - Targets",
                            "model_type: ModelType - Type of model",
                            "test_size: float = 0.2 - Test set proportion"
                        ],
                        "returns": "EvaluationResult - Comprehensive evaluation result",
                        "workflow": [
                            "1. Split data into train/test sets",
                            "2. Train model if not already fitted",
                            "3. Generate predictions",
                            "4. Apply appropriate evaluator based on model type",
                            "5. Perform cross-validation",
                            "6. Extract feature importance if available",
                            "7. Package results in EvaluationResult"
                        ],
                        "features": [
                            "Automatic train/test splitting",
                            "Model fitting detection",
                            "Type-specific evaluation",
                            "Integrated validation",
                            "Comprehensive result packaging"
                        ]
                    }
                },
                "usage_example": '''
# Initialize evaluation suite
suite = EvaluationSuite()

# Comprehensive model evaluation
result = suite.evaluate_model(
    model=my_model,
    X=features,
    y=targets,
    model_type=ModelType.CLASSIFICATION,
    test_size=0.2
)

# Access results
print(f"Model: {result.model_name}")
print(f"Metrics: {result.metrics}")
print(f"CV Scores: {result.cross_validation_scores}")
print(result.classification_report)
'''
            }
        }
    
    def get_module_overview(self) -> str:
        """Generate comprehensive module overview."""
        overview = f"""
{'='*80}
MODEL EVALUATION MODULE OVERVIEW
{'='*80}

Module: {self.module_info.name}
Version: {self.module_info.version}
Last Updated: {self.module_info.last_updated}

DESCRIPTION:
{self.module_info.description}

The Model Evaluation Module provides a comprehensive framework for evaluating
machine learning models across different problem types. It offers standardized
interfaces, extensive metrics, advanced validation techniques, and professional
visualizations.

CORE CAPABILITIES:
"""
        for i, capability in enumerate(self.module_info.capabilities, 1):
            overview += f"{i:2d}. {capability}\n"
        
        overview += f"""
MAIN DEPENDENCIES:
{', '.join(self.module_info.dependencies)}

ARCHITECTURE:
The module follows a hierarchical design pattern:

1. BASE LAYER (BaseEvaluator):
   - Abstract base class defining evaluation interface
   - Common functionality and utilities
   - Standardized method signatures

2. SPECIALIZED EVALUATORS:
   - RegressionEvaluator: Regression-specific metrics and visualizations
   - ClassificationEvaluator: Classification metrics with ROC/PR curves
   - Future extensions for other model types

3. VALIDATION LAYER (ModelValidator):
   - Cross-validation capabilities
   - Validation and learning curves
   - Advanced validation techniques

4. INTEGRATION LAYER (EvaluationSuite):
   - Unified evaluation interface
   - Automatic workflow orchestration
   - Comprehensive result packaging

5. DATA STRUCTURES:
   - EvaluationResult: Standardized result container
   - MetricType: Evaluation type categorization
   - Type-safe interfaces with full typing support

KEY DESIGN PRINCIPLES:
- Modularity: Each component serves a specific purpose
- Extensibility: Easy to add new evaluators and metrics
- Consistency: Standardized interfaces across all evaluators
- Flexibility: Configurable metrics and parameters
- Robustness: Comprehensive error handling and validation
- Usability: Clear APIs with extensive documentation

INTEGRATION POINTS:
- Base modeling framework (ModelType, ProblemType)
- Visualization libraries (matplotlib, seaborn)
- Scientific computing stack (numpy, pandas, scipy)
- Machine learning library (scikit-learn)

{'='*80}
"""
        return overview
    
    def get_component_details(self, component_type: str = "all") -> str:
        """
        Get detailed information about specific component types.
        
        Args:
            component_type: Type of components to detail ("evaluators", "validators", "suites", "all")
        """
        if not IMPORTS_AVAILABLE:
            return "Component analysis unavailable - imports failed"
        
        details = f"\n{'='*80}\nCOMPONENT DETAILED ANALYSIS\n{'='*80}\n"
        
        components_to_show = []
        if component_type == "all":
            components_to_show = ["enums", "data_classes", "evaluators", "validators", "suites"]
        else:
            components_to_show = [component_type] if component_type in self.module_info.components else []
        
        for comp_type in components_to_show:
            if comp_type in self.module_info.components:
                details += self._format_component_section(comp_type, self.module_info.components[comp_type])
        
        return details
    
    def _format_component_section(self, section_name: str, components: Dict[str, Any]) -> str:
        """Format a component section for display."""
        section = f"\n{section_name.upper().replace('_', ' ')}:\n{'-'*60}\n"
        
        for comp_name, comp_info in components.items():
            section += f"\n{comp_name}:\n"
            section += f"  Description: {comp_info.get('description', 'N/A')}\n"
            
            if 'purpose' in comp_info:
                section += f"  Purpose: {comp_info['purpose']}\n"
            
            if 'inheritance' in comp_info:
                section += f"  Inherits from: {comp_info['inheritance']}\n"
            
            if 'key_features' in comp_info:
                section += "  Key Features:\n"
                for feature in comp_info['key_features']:
                    section += f"    • {feature}\n"
            
            if 'values' in comp_info:
                section += "  Values:\n"
                for value in comp_info['values']:
                    section += f"    • {value}\n"
            
            if 'fields' in comp_info:
                section += "  Fields:\n"
                for field in comp_info['fields']:
                    section += f"    • {field}\n"
            
            if 'methods' in comp_info:
                section += "  Methods:\n"
                for method_name, method_info in comp_info['methods'].items():
                    section += f"    {method_name}():\n"
                    section += f"      - {method_info.get('description', 'N/A')}\n"
                    
                    if 'parameters' in method_info:
                        section += "      Parameters:\n"
                        for param in method_info['parameters']:
                            section += f"        • {param}\n"
                    
                    if 'returns' in method_info:
                        section += f"      Returns: {method_info['returns']}\n"
                    
                    if 'supported_metrics' in method_info:
                        section += "      Supported Metrics:\n"
                        for metric in method_info['supported_metrics']:
                            section += f"        • {metric}\n"
                    
                    if 'features' in method_info:
                        section += "      Features:\n"
                        for feature in method_info['features']:
                            section += f"        • {feature}\n"
            
            if 'usage_example' in comp_info:
                section += f"  Usage Example:\n{comp_info['usage_example']}\n"
            
            section += "\n"
        
        return section
    
    def get_usage_examples(self) -> str:
        """Generate comprehensive usage examples."""
        examples = f"""
{'='*80}
COMPREHENSIVE USAGE EXAMPLES
{'='*80}

1. REGRESSION MODEL EVALUATION:
{'-'*40}

from modeling.evaluation import RegressionEvaluator, EvaluationSuite
import numpy as np

# Sample data
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 1.9, 3.1, 3.8, 5.2])

# Basic regression evaluation
reg_evaluator = RegressionEvaluator(['mse', 'mae', 'r2', 'rmse'])
metrics = reg_evaluator.evaluate(y_true, y_pred)
print(f"Regression Metrics: {{metrics}}")

# Generate comprehensive report
report = reg_evaluator.generate_report(y_true, y_pred)
print(report)

# Create visualizations
reg_evaluator.plot_predictions(y_true, y_pred, 'regression_analysis.png')

2. CLASSIFICATION MODEL EVALUATION:
{'-'*40}

from modeling.evaluation import ClassificationEvaluator
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Advanced classification evaluation
clf_evaluator = ClassificationEvaluator(
    metrics=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'],
    average='weighted'
)

# Evaluate with probabilities
metrics = clf_evaluator.evaluate(y_test, y_pred, y_proba)
print(f"Classification Metrics: {{metrics}}")

# Generate detailed report
report = clf_evaluator.generate_report(
    y_test, y_pred, 
    target_names=['Class_0', 'Class_1', 'Class_2'],
    model_name="Random Forest Classifier"
)
print(report)

# Visualizations
clf_evaluator.plot_confusion_matrix(y_test, y_pred, ['Class_0', 'Class_1', 'Class_2'])
roc_data = clf_evaluator.plot_roc_curves(y_test, y_proba, ['Class_0', 'Class_1', 'Class_2'])
pr_data = clf_evaluator.plot_precision_recall_curves(y_test, y_proba, ['Class_0', 'Class_1', 'Class_2'])

3. ADVANCED MODEL VALIDATION:
{'-'*40}

from modeling.evaluation import ModelValidator
from sklearn.linear_model import LogisticRegression

# Initialize validator
validator = ModelValidator(cv=5, random_state=42)

# Cross-validation
cv_scores = validator.cross_validate(
    model, X_train, y_train, 
    scoring=['accuracy', 'f1_weighted', 'precision_weighted']
)
print(f"Cross-validation scores: {{cv_scores}}")

# Validation curve analysis
param_range = [0.01, 0.1, 1, 10, 100]
param_vals, train_scores, val_scores = validator.validation_curve_analysis(
    LogisticRegression(max_iter=1000), X_train, y_train,
    param_name='C', param_range=param_range, scoring='accuracy'
)

# Plot validation curve
validator.plot_validation_curve(
    param_vals, train_scores, val_scores, 'C', 'validation_curve.png'
)

# Learning curve analysis
train_sizes, train_scores, val_scores = validator.learning_curve_analysis(
    model, X_train, y_train, scoring='accuracy'
)

# Plot learning curve
validator.plot_learning_curve(
    train_sizes, train_scores, val_scores, 'learning_curve.png'
)

4. COMPREHENSIVE EVALUATION SUITE:
{'-'*40}

from modeling.evaluation import EvaluationSuite
from modeling.base import ModelType

# Initialize evaluation suite
suite = EvaluationSuite()

# Comprehensive evaluation
result = suite.evaluate_model(
    model=model,
    X=X,
    y=y,
    model_type=ModelType.CLASSIFICATION,
    test_size=0.2
)

# Access comprehensive results
print(f"Model Name: {{result.model_name}}")
print(f"Metrics: {{result.metrics}}")
print(f"Cross-validation Scores: {{result.cross_validation_scores}}")
print(result.classification_report)

if result.feature_importance:
    print(f"Feature Importance: {{result.feature_importance}}")

5. CUSTOM METRICS AND CONFIGURATION:
{'-'*40}

# Custom regression evaluator
custom_reg_evaluator = RegressionEvaluator([
    'mse', 'mae', 'r2', 'rmse', 'mape', 'max_error', 'explained_variance'
])

# Custom classification evaluator with specific averaging
custom_clf_evaluator = ClassificationEvaluator(
    metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'log_loss'],
    average='macro'  # Use macro averaging for multiclass
)

# Custom validator with different CV strategy
custom_validator = ModelValidator(cv=10, random_state=123)

6. BINARY CLASSIFICATION SPECIFIC:
{'-'*40}

from sklearn.datasets import make_classification

# Binary classification data
X_bin, y_bin = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_bin, y_bin, test_size=0.2, random_state=42
)

# Binary classifier
binary_model = LogisticRegression(random_state=42)
binary_model.fit(X_train_bin, y_train_bin)

y_pred_bin = binary_model.predict(X_test_bin)
y_proba_bin = binary_model.predict_proba(X_test_bin)

# Binary-specific evaluation
binary_evaluator = ClassificationEvaluator(
    metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    average='binary'
)

metrics_bin = binary_evaluator.evaluate(y_test_bin, y_pred_bin, y_proba_bin)
print(f"Binary Classification Metrics: {{metrics_bin}}")

# Binary-specific visualizations
binary_evaluator.plot_confusion_matrix(y_test_bin, y_pred_bin, ['Negative', 'Positive'])
roc_data_bin = binary_evaluator.plot_roc_curves(y_test_bin, y_proba_bin, ['Negative', 'Positive'])

{'='*80}
"""
        return examples
    
    def get_best_practices(self) -> str:
        """Generate best practices guide."""
        return f"""
{'='*80}
BEST PRACTICES AND GUIDELINES
{'='*80}

1. EVALUATOR SELECTION:
{'-'*30}
• Use RegressionEvaluator for continuous target variables
• Use ClassificationEvaluator for categorical target variables  
• Choose appropriate averaging strategy for multiclass problems:
  - 'weighted': Best for imbalanced datasets
  - 'macro': Equal weight to all classes
  - 'micro': Global metric across all samples
• Consider problem-specific metrics (e.g., F1 for imbalanced classes)

2. METRIC CONFIGURATION:
{'-'*30}
• Start with default metrics for baseline evaluation
• Add specialized metrics based on business requirements:
  - MAPE for interpretable percentage errors in regression
  - ROC AUC for ranking/probability assessment
  - Precision/Recall for imbalanced classification
• Always include cross-validation for robust estimates

3. VALIDATION STRATEGIES:
{'-'*30}
• Use 5-10 fold cross-validation for small to medium datasets
• Use stratified CV for classification to maintain class distribution
• Generate validation curves when tuning hyperparameters
• Create learning curves to assess training data sufficiency
• Set random_state for reproducible results

4. VISUALIZATION BEST PRACTICES:
{'-'*30}
• Always save plots for documentation and reporting
• Use descriptive titles and class names in visualizations
• Include confidence intervals in validation/learning curves
• Normalize confusion matrices for better comparison
• Create ROC curves for binary and multiclass problems

5. PERFORMANCE OPTIMIZATION:
{'-'*30}
• Use sparse matrix handling for high-dimensional data
• Implement early stopping for expensive evaluations
• Cache evaluation results for repeated analysis
• Use parallel processing in cross-validation when possible

6. ERROR HANDLING:
{'-'*30}
• Always check for missing or invalid predictions
• Handle edge cases (empty predictions, single class)
• Validate input shapes and types before evaluation
• Provide meaningful error messages and fallbacks

7. REPORTING AND DOCUMENTATION:
{'-'*30}
• Generate comprehensive reports for stakeholders
• Include both overall and class-wise metrics
• Document evaluation methodology and parameters
• Provide context for metric interpretation
• Save evaluation artifacts for reproducibility

8. INTEGRATION PATTERNS:
{'-'*30}
• Use EvaluationSuite for standardized workflows
• Integrate with model training pipelines
• Store evaluation results in structured formats
• Version control evaluation configurations
• Automate evaluation in CI/CD pipelines

9. COMMON PITFALLS TO AVOID:
{'-'*30}
• Don't use accuracy alone for imbalanced datasets
• Avoid data leakage in cross-validation splits
• Don't ignore confidence intervals in performance estimates
• Avoid comparing models without proper statistical testing
• Don't forget to evaluate on truly held-out test sets

10. ADVANCED TECHNIQUES:
{'-'*30}
• Use nested cross-validation for unbiased hyperparameter tuning
• Implement custom scoring functions for domain-specific metrics
• Combine multiple evaluation approaches for comprehensive assessment
• Use bootstrap sampling for robust confidence intervals
• Implement statistical significance testing for model comparison

{'='*80}
"""

    def generate_complete_documentation(self) -> str:
        """Generate complete module documentation."""
        doc = self.get_module_overview()
        doc += self.get_component_details("all")
        doc += self.get_usage_examples()
        doc += self.get_best_practices()
        
        doc += f"""
{'='*80}
MODULE SUMMARY
{'='*80}

The Model Evaluation Module provides a comprehensive, production-ready
framework for evaluating machine learning models. Key strengths include:

TECHNICAL EXCELLENCE:
• Type-safe interfaces with comprehensive typing
• Robust error handling and validation
• Efficient sparse matrix support
• Professional visualization capabilities
• Extensive metric coverage

ARCHITECTURAL DESIGN:
• Modular, extensible component architecture
• Standardized interfaces across evaluators
• Clean separation of concerns
• Integration-ready design patterns

USABILITY FEATURES:
• Intuitive APIs with sensible defaults
• Comprehensive documentation and examples
• Flexible configuration options
• Professional reporting capabilities

PRODUCTION READINESS:
• Thorough testing and validation
• Performance optimization
• Comprehensive logging
• Integration support

The module serves as a foundational component for any machine learning
evaluation workflow, providing the tools necessary for rigorous model
assessment and validation.

{'='*80}
END OF DOCUMENTATION
{'='*80}
"""
        
        return doc


def main():
    """Main function to demonstrate the evaluation module analysis."""
    print("Model Evaluation Module Information System")
    print("=" * 50)
    
    analyzer = EvaluationModuleAnalyzer()
    
    # Generate and display complete documentation
    complete_doc = analyzer.generate_complete_documentation()
    
    # For demonstration, show just the overview
    print(analyzer.get_module_overview())
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()
