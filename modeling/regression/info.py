"""
Regression Module Information Module.

This module provides comprehensive information about the regression module
capabilities, features, and usage guidelines for supervised regression methods
including linear regression, polynomial regression, and regularized regression.
"""

import logging
from typing import Dict, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)


def get_package_info() -> Dict[str, Any]:
    """
    Get comprehensive regression module information.
    
    Returns:
        Dictionary containing complete module details
    """
    return {
        'module_name': 'Advanced Regression Framework',
        'version': '1.0.0',
        'description': 'Comprehensive regression framework with classical and modern supervised learning algorithms for continuous target prediction',
        'author': 'AI ML Platform Team',
        'license': 'MIT',
        'created_date': '2025-07-27',
        'last_updated': datetime.now().isoformat(),
        'core_components': {
            'regression_models': {
                'file': '__init__.py',
                'lines_of_code': 561,
                'description': 'Advanced regression models with comprehensive evaluation and interpretation capabilities',
                'key_classes': ['LinearRegressionModel', 'PolynomialRegressionModel', 'RegressionModelFactory'],
                'features': [
                    'Linear regression with optional regularization (Ridge, Lasso, ElasticNet)',
                    'Polynomial regression for non-linear relationships',
                    'Automatic model evaluation with comprehensive metrics',
                    'Cross-validation and model selection',
                    'Hyperparameter optimization support',
                    'Model persistence and deployment'
                ]
            },
            'base_integration': {
                'file': '../base/__init__.py',
                'description': 'BaseModel interface compliance for consistent API',
                'key_classes': ['BaseModel', 'ModelMetadata', 'ModelType', 'ProblemType', 'ModelStatus'],
                'features': [
                    'Standardized fit/predict interface',
                    'Model lifecycle management',
                    'Metadata tracking and versioning',
                    'Status monitoring and error handling',
                    'Type-safe implementations'
                ]
            }
        },
        'supported_algorithms': {
            'linear_models': {
                'linear_regression': {
                    'description': 'Linear regression for continuous target prediction',
                    'class_name': 'LinearRegressionModel',
                    'algorithm_type': 'Linear regression',
                    'strengths': ['Fast training', 'Interpretable coefficients', 'No hyperparameter tuning needed'],
                    'weaknesses': ['Assumes linear relationships', 'Sensitive to outliers'],
                    'best_use_cases': ['Baseline models', 'Large datasets', 'Interpretable models'],
                    'hyperparameters': {
                        'fit_intercept': 'bool (whether to fit intercept)',
                        'normalize': 'bool (whether to normalize features)',
                        'regularization': ['None', 'ridge', 'lasso', 'elastic_net'],
                        'alpha': 'float (regularization strength)'
                    },
                    'complexity': 'O(n × p) training, O(p) prediction',
                    'output_types': ['predicted_values', 'coefficients']
                }
            },
            'polynomial_models': {
                'polynomial_regression': {
                    'description': 'Polynomial regression for non-linear relationships',
                    'class_name': 'PolynomialRegressionModel',
                    'algorithm_type': 'Polynomial regression',
                    'strengths': ['Captures non-linear relationships', 'Flexible feature transformations'],
                    'weaknesses': ['Prone to overfitting', 'High computational cost for large degrees'],
                    'best_use_cases': ['Non-linear patterns', 'Feature engineering'],
                    'hyperparameters': {
                        'degree': 'int (degree of polynomial features)',
                        'include_bias': 'bool (whether to include bias column)',
                        'interaction_only': 'bool (whether to include only interaction features)',
                        'regularization': ['None', 'ridge', 'lasso', 'elastic_net'],
                        'alpha': 'float (regularization strength)'
                    },
                    'complexity': 'O(n × p × degree) training, O(p × degree) prediction',
                    'output_types': ['predicted_values', 'polynomial_coefficients']
                }
            }
        },
        'problem_types_supported': {
            'linear_regression': {
                'description': 'Predict continuous target values using linear relationships',
                'examples': ['House price prediction', 'Stock price forecasting', 'Sales prediction'],
                'metrics': ['Mean Squared Error (MSE)', 'Mean Absolute Error (MAE)', 'R² Score'],
                'special_considerations': ['Feature scaling for regularized models', 'Outlier detection'],
                'output_interpretation': 'Continuous target values'
            },
            'polynomial_regression': {
                'description': 'Predict continuous target values using polynomial relationships',
                'examples': ['Non-linear trend analysis', 'Physics simulations', 'Complex curve fitting'],
                'metrics': ['Mean Squared Error (MSE)', 'Mean Absolute Error (MAE)', 'R² Score'],
                'special_considerations': ['Degree selection to balance bias-variance tradeoff', 'Feature scaling for regularized models'],
                'output_interpretation': 'Continuous target values'
            }
        },
        'evaluation_framework': {
            'performance_metrics': {
                'mse': {
                    'description': 'Mean Squared Error',
                    'formula': 'Σ(y_true - y_pred)² / n',
                    'range': '[0, ∞] where 0 is perfect',
                    'best_for': 'General-purpose regression evaluation'
                },
                'mae': {
                    'description': 'Mean Absolute Error',
                    'formula': 'Σ|y_true - y_pred| / n',
                    'range': '[0, ∞] where 0 is perfect',
                    'best_for': 'Interpretability and robustness to outliers'
                },
                'r2': {
                    'description': 'R² Score',
                    'formula': '1 - (Σ(y_true - y_pred)² / Σ(y_true - mean(y_true))²)',
                    'range': '(-∞, 1] where 1 is perfect',
                    'best_for': 'Explained variance assessment'
                },
                'rmse': {
                    'description': 'Root Mean Squared Error',
                    'formula': 'sqrt(MSE)',
                    'range': '[0, ∞] where 0 is perfect',
                    'best_for': 'General-purpose regression evaluation'
                }
            },
            'model_interpretation': {
                'coefficients': 'Linear regression coefficients for feature importance',
                'polynomial_coefficients': 'Polynomial regression coefficients for feature importance',
                'residual_analysis': 'Analyze residuals to assess model fit',
                'adjusted_r2': 'Adjusted R² for model complexity penalty'
            }
        }
    }


def export_info_json(filename: str = 'regression_info.json') -> None:
    """Export complete module information to JSON file."""
    info_data = {
        'package_info': get_package_info(),
        'generated_at': datetime.now().isoformat()
    }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(info_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Regression module information exported to {filename}")
        print(f"✅ Module information exported to {filename}")
    except Exception as e:
        logger.error(f"Failed to export module information: {e}")
        print(f"❌ Export failed: {e}")


if __name__ == "__main__":
    # Generate and display summary when run directly
    print("🎯 Regression Module Information")
    print("=" * 60)
    print(json.dumps(get_package_info(), indent=2))
    print("\n" + "=" * 60)
    
    # Optionally export to JSON
    export_choice = input("\nExport detailed information to JSON? (y/n): ").lower().strip()
    if export_choice in ['y', 'yes']:
        export_info_json()
    
    print("\n📚 Documentation complete!")
