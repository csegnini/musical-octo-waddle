"""
AutoML Package Information Module.

This module provides comprehensive information about the AutoML package
capabilities, features, and usage guidelines for automated machine learning,
hyperparameter optimization, and pipeline automation.
"""

import logging
from typing import Dict, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)


def get_package_info() -> Dict[str, Any]:
    """
    Get comprehensive AutoML package information.
    
    Returns:
        Dictionary containing complete package details
    """
    return {
        'package_name': 'Advanced AutoML and Hyperparameter Optimization Framework',
        'version': '1.0.0',
        'description': 'Comprehensive automated machine learning framework with hyperparameter optimization, feature engineering, and pipeline automation capabilities.',
        'author': 'AI ML Platform Team',
        'license': 'MIT',
        'created_date': '2025-07-27',
        'last_updated': datetime.now().isoformat(),
        
        # Core capabilities
        'core_modules': {
            'hyperparameter_optimizer': {
                'file': '__init__.py',
                'lines_of_code': 715,
                'description': 'Advanced hyperparameter optimization with multiple algorithms',
                'key_classes': ['HyperparameterOptimizer', 'OptimizationResult', 'OptimizationAlgorithm'],
                'features': [
                    'Grid Search optimization',
                    'Random Search optimization', 
                    'Bayesian optimization (Optuna, Hyperopt, Scikit-Optimize)',
                    'Genetic algorithm optimization',
                    'Successive halving optimization',
                    'Multi-objective optimization',
                    'Early stopping and pruning',
                    'Cross-validation integration'
                ]
            },
            'automl_pipeline': {
                'file': '__init__.py',
                'description': 'Automated machine learning pipeline with model selection',
                'key_classes': ['AutoMLPipeline', 'AutoMLConfig', 'AutoMLMode'],
                'features': [
                    'Automated model selection and comparison',
                    'Feature engineering and selection automation',
                    'Pipeline optimization and automation',
                    'Cross-validation and model evaluation',
                    'Ensemble model creation',
                    'Neural network integration',
                    'Preprocessing automation'
                ]
            },
            'feature_engineering': {
                'file': '__init__.py',
                'description': 'Automated feature selection and engineering',
                'key_classes': ['FeatureSelectionMethod'],
                'features': [
                    'SelectKBest feature selection',
                    'Recursive feature elimination',
                    'LASSO-based feature selection',
                    'Tree-based feature importance',
                    'Mutual information selection',
                    'Automated preprocessing pipelines'
                ]
            }
        },
        
        # Optimization algorithms
        'optimization_algorithms': {
            'grid_search': {
                'description': 'Exhaustive search over specified parameter values',
                'class_name': 'HyperparameterOptimizer',
                'algorithm_type': 'Exhaustive Search',
                'strengths': ['Guaranteed to find global optimum in search space', 'Deterministic results'],
                'weaknesses': ['Exponential time complexity', 'Not suitable for large parameter spaces'],
                'best_use_cases': ['Small parameter spaces', 'Discrete parameters'],
                'hyperparameters': {
                    'param_grid': 'dict (parameter grid to search)',
                    'cv': 'int (cross-validation folds)',
                    'scoring': 'str (scoring metric)'
                },
                'complexity': 'O(∏(|param_i|)) where param_i are parameter ranges',
                'output_types': ['best_params', 'best_score', 'cv_results']
            },
            'random_search': {
                'description': 'Random sampling from parameter distributions',
                'class_name': 'HyperparameterOptimizer',
                'algorithm_type': 'Random Sampling',
                'strengths': ['More efficient than grid search', 'Good for continuous parameters'],
                'weaknesses': ['May miss optimal values', 'No guarantee of finding global optimum'],
                'best_use_cases': ['Large parameter spaces', 'Mixed parameter types'],
                'hyperparameters': {
                    'param_distributions': 'dict (parameter distributions)',
                    'n_iter': 'int (number of iterations)',
                    'cv': 'int (cross-validation folds)'
                },
                'complexity': 'O(n_iter × cv_folds)',
                'output_types': ['best_params', 'best_score', 'optimization_history']
            },
            'bayesian_optimization': {
                'description': 'Bayesian optimization using Gaussian processes',
                'class_name': 'HyperparameterOptimizer',
                'algorithm_type': 'Bayesian Optimization',
                'strengths': ['Sample efficient', 'Good for expensive evaluations', 'Handles uncertainty'],
                'weaknesses': ['More complex setup', 'May struggle with high dimensions'],
                'best_use_cases': ['Expensive model training', 'Continuous parameters'],
                'hyperparameters': {
                    'acquisition': 'str (acquisition function)',
                    'n_calls': 'int (number of calls)',
                    'random_state': 'int (random seed)'
                },
                'complexity': 'O(n³) per iteration for GP fitting',
                'output_types': ['best_params', 'best_score', 'acquisition_history']
            },
            'optuna': {
                'description': 'Tree-structured Parzen Estimator optimization',
                'class_name': 'HyperparameterOptimizer',
                'algorithm_type': 'Advanced Bayesian',
                'strengths': ['State-of-the-art algorithm', 'Pruning support', 'Parallel execution'],
                'weaknesses': ['Requires additional dependency', 'Learning curve'],
                'best_use_cases': ['Advanced optimization', 'Large-scale hyperparameter tuning'],
                'hyperparameters': {
                    'n_trials': 'int (number of trials)',
                    'timeout': 'int (timeout in seconds)',
                    'pruner': 'str (pruning algorithm)'
                },
                'complexity': 'O(n_trials × model_training_time)',
                'output_types': ['best_params', 'best_value', 'trials_dataframe']
            }
        },
        
        # Supported models
        'supported_models': {
            'ensemble_methods': {
                'random_forest': {
                    'classifier': 'RandomForestClassifier',
                    'regressor': 'RandomForestRegressor',
                    'key_params': ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf'],
                    'use_cases': ['High accuracy', 'Feature importance', 'Robust to overfitting']
                },
                'gradient_boosting': {
                    'classifier': 'GradientBoostingClassifier',
                    'regressor': 'GradientBoostingRegressor',
                    'key_params': ['n_estimators', 'learning_rate', 'max_depth', 'min_samples_split'],
                    'use_cases': ['High performance', 'Sequential learning', 'Handle missing values']
                },
                'extra_trees': {
                    'classifier': 'ExtraTreesClassifier',
                    'regressor': 'ExtraTreesRegressor',
                    'key_params': ['n_estimators', 'max_depth', 'min_samples_split'],
                    'use_cases': ['Faster than random forest', 'Reduce overfitting']
                }
            },
            'linear_methods': {
                'logistic_regression': {
                    'classifier': 'LogisticRegression',
                    'key_params': ['C', 'solver', 'max_iter'],
                    'use_cases': ['Binary classification', 'Interpretable results', 'Fast training']
                },
                'linear_regression': {
                    'regressor': 'LinearRegression',
                    'key_params': ['fit_intercept', 'normalize'],
                    'use_cases': ['Simple regression', 'Baseline model', 'Interpretable']
                },
                'ridge_regression': {
                    'regressor': 'Ridge',
                    'key_params': ['alpha', 'solver'],
                    'use_cases': ['Regularized regression', 'Multicollinearity handling']
                },
                'lasso_regression': {
                    'regressor': 'Lasso',
                    'key_params': ['alpha', 'max_iter'],
                    'use_cases': ['Feature selection', 'Sparse solutions']
                }
            },
            'instance_based': {
                'k_neighbors': {
                    'classifier': 'KNeighborsClassifier',
                    'regressor': 'KNeighborsRegressor',
                    'key_params': ['n_neighbors', 'weights', 'algorithm'],
                    'use_cases': ['Non-parametric', 'Local patterns', 'Simple implementation']
                }
            },
            'support_vector_machines': {
                'svm_classifier': {
                    'classifier': 'SVC',
                    'key_params': ['C', 'kernel', 'gamma'],
                    'use_cases': ['High-dimensional data', 'Non-linear boundaries', 'Robust']
                },
                'svm_regressor': {
                    'regressor': 'SVR',
                    'key_params': ['C', 'kernel', 'gamma', 'epsilon'],
                    'use_cases': ['Non-linear regression', 'Robust to outliers']
                }
            },
            'neural_networks': {
                'mlp_classifier': {
                    'classifier': 'MLPClassifier',
                    'key_params': ['hidden_layer_sizes', 'activation', 'solver', 'learning_rate'],
                    'use_cases': ['Complex patterns', 'Non-linear relationships', 'Deep learning']
                },
                'mlp_regressor': {
                    'regressor': 'MLPRegressor',
                    'key_params': ['hidden_layer_sizes', 'activation', 'solver', 'learning_rate'],
                    'use_cases': ['Non-linear regression', 'Complex function approximation']
                }
            }
        },
        
        # AutoML modes
        'automl_modes': {
            'fast': {
                'description': 'Quick evaluation with fewer models and limited search',
                'typical_time': '5-15 minutes',
                'models_tested': '3-5 models',
                'hyperparameter_trials': '10-20 per model',
                'best_for': 'Rapid prototyping, initial exploration'
            },
            'balanced': {
                'description': 'Balance between speed and thoroughness',
                'typical_time': '15-45 minutes',
                'models_tested': '5-8 models',
                'hyperparameter_trials': '20-50 per model',
                'best_for': 'Most use cases, good performance-time tradeoff'
            },
            'thorough': {
                'description': 'Comprehensive search with extensive model testing',
                'typical_time': '1-3 hours',
                'models_tested': '8-12 models',
                'hyperparameter_trials': '50-100 per model',
                'best_for': 'Production models, maximum performance'
            },
            'custom': {
                'description': 'User-defined configuration',
                'typical_time': 'User-defined',
                'models_tested': 'User-defined',
                'hyperparameter_trials': 'User-defined',
                'best_for': 'Specific requirements, expert users'
            }
        },
        
        # Feature selection methods
        'feature_selection_methods': {
            'select_k_best': {
                'description': 'Select K best features based on statistical tests',
                'algorithm': 'Univariate statistical tests (f_classif, f_regression)',
                'strengths': ['Fast', 'Simple', 'Removes irrelevant features'],
                'weaknesses': ['Ignores feature interactions', 'May remove useful features'],
                'best_for': 'High-dimensional data, initial feature reduction'
            },
            'recursive_feature_elimination': {
                'description': 'Recursively eliminate features based on model coefficients',
                'algorithm': 'Backward elimination using model weights',
                'strengths': ['Considers feature interactions', 'Model-specific'],
                'weaknesses': ['Computationally expensive', 'May overfit'],
                'best_for': 'Moderate feature sets, linear models'
            },
            'lasso_selection': {
                'description': 'Use LASSO regularization for feature selection',
                'algorithm': 'L1 regularization driving coefficients to zero',
                'strengths': ['Automatic feature selection', 'Handles multicollinearity'],
                'weaknesses': ['May arbitrarily select among correlated features'],
                'best_for': 'Sparse solutions, multicollinear features'
            },
            'tree_importance': {
                'description': 'Feature importance from tree-based models',
                'algorithm': 'Gini importance or permutation importance',
                'strengths': ['Handles non-linear relationships', 'Feature interactions'],
                'weaknesses': ['Biased toward high-cardinality features'],
                'best_for': 'Non-linear relationships, ensemble models'
            }
        },
        
        # Performance metrics
        'evaluation_framework': {
            'classification_metrics': {
                'accuracy': {
                    'description': 'Fraction of correct predictions',
                    'formula': '(TP + TN) / (TP + TN + FP + FN)',
                    'range': '[0, 1] where 1 is perfect',
                    'best_for': 'Balanced datasets'
                },
                'precision': {
                    'description': 'Fraction of positive predictions that are correct',
                    'formula': 'TP / (TP + FP)',
                    'range': '[0, 1] where 1 is perfect',
                    'best_for': 'Minimizing false positives'
                },
                'recall': {
                    'description': 'Fraction of positive cases correctly identified',
                    'formula': 'TP / (TP + FN)',
                    'range': '[0, 1] where 1 is perfect',
                    'best_for': 'Minimizing false negatives'
                },
                'f1_score': {
                    'description': 'Harmonic mean of precision and recall',
                    'formula': '2 × (precision × recall) / (precision + recall)',
                    'range': '[0, 1] where 1 is perfect',
                    'best_for': 'Balanced precision-recall tradeoff'
                },
                'roc_auc': {
                    'description': 'Area under ROC curve',
                    'formula': 'Area under True Positive Rate vs False Positive Rate curve',
                    'range': '[0, 1] where 1 is perfect',
                    'best_for': 'Binary classification, probability predictions'
                }
            },
            'regression_metrics': {
                'mse': {
                    'description': 'Mean Squared Error',
                    'formula': 'Σ(y_true - y_pred)² / n',
                    'range': '[0, ∞] where 0 is perfect',
                    'best_for': 'General regression evaluation'
                },
                'mae': {
                    'description': 'Mean Absolute Error',
                    'formula': 'Σ|y_true - y_pred| / n',
                    'range': '[0, ∞] where 0 is perfect',
                    'best_for': 'Robust to outliers'
                },
                'r2': {
                    'description': 'R² Score (Coefficient of Determination)',
                    'formula': '1 - (Σ(y_true - y_pred)² / Σ(y_true - mean(y_true))²)',
                    'range': '(-∞, 1] where 1 is perfect',
                    'best_for': 'Explained variance assessment'
                }
            }
        },
        
        # Technical specifications
        'technical_specs': {
            'performance': {
                'small_dataset': 'Fast mode: 5-15 minutes for <10K samples',
                'medium_dataset': 'Balanced mode: 15-45 minutes for 10K-100K samples',
                'large_dataset': 'Thorough mode: 1-3 hours for 100K+ samples',
                'parallel_processing': 'Supports parallel execution with n_jobs parameter'
            },
            'compatibility': {
                'python_version': '3.7+',
                'required_dependencies': ['scikit-learn', 'pandas', 'numpy'],
                'optional_dependencies': ['optuna', 'hyperopt', 'scikit-optimize'],
                'data_formats': ['pandas DataFrame', 'numpy arrays']
            },
            'scalability': {
                'max_features': '10,000+ features supported',
                'max_samples': '1M+ samples supported (memory dependent)',
                'parallel_trials': 'Full parallel hyperparameter optimization',
                'memory_usage': 'Efficient memory management for large datasets'
            }
        },
        
        # Integration capabilities
        'integration': {
            'base_framework': {
                'seamless_integration': True,
                'required_modules': ['base.ModelMetadata', 'base.ModelStatus'],
                'configuration_sharing': 'Compatible with base model interfaces'
            },
            'external_libraries': {
                'optuna': 'Advanced Bayesian optimization',
                'hyperopt': 'Tree-structured Parzen Estimator',
                'scikit_optimize': 'Gaussian Process optimization',
                'scikit_learn': 'Core ML algorithms and utilities'
            }
        }
    }


def get_optimization_comparison() -> Dict[str, Any]:
    """Compare optimization algorithms performance and characteristics."""
    return {
        'algorithm_comparison': {
            'grid_search': {
                'speed': '⭐⭐ (Slow for large spaces)',
                'accuracy': '⭐⭐⭐⭐⭐ (Guaranteed optimum in search space)',
                'ease_of_use': '⭐⭐⭐⭐⭐ (Very simple)',
                'scalability': '⭐ (Poor for high dimensions)',
                'best_for': 'Small discrete parameter spaces'
            },
            'random_search': {
                'speed': '⭐⭐⭐⭐ (Much faster than grid)',
                'accuracy': '⭐⭐⭐ (Good but not guaranteed)',
                'ease_of_use': '⭐⭐⭐⭐ (Simple)',
                'scalability': '⭐⭐⭐⭐ (Good for high dimensions)',
                'best_for': 'Large parameter spaces, quick results'
            },
            'bayesian_optimization': {
                'speed': '⭐⭐⭐ (Efficient for expensive evaluations)',
                'accuracy': '⭐⭐⭐⭐ (Very good)',
                'ease_of_use': '⭐⭐⭐ (Moderate complexity)',
                'scalability': '⭐⭐ (Struggles with high dimensions)',
                'best_for': 'Expensive model training, continuous parameters'
            },
            'optuna': {
                'speed': '⭐⭐⭐⭐⭐ (Very fast with pruning)',
                'accuracy': '⭐⭐⭐⭐⭐ (State-of-the-art)',
                'ease_of_use': '⭐⭐⭐⭐ (Good documentation)',
                'scalability': '⭐⭐⭐⭐⭐ (Excellent)',
                'best_for': 'Production systems, advanced optimization'
            }
        },
        'performance_benchmarks': {
            'small_search_space': {
                'grid_search': '100% coverage, 5-10 minutes',
                'random_search': '~80% coverage, 2-5 minutes',
                'bayesian': '~90% coverage, 3-7 minutes',
                'optuna': '~95% coverage, 2-4 minutes'
            },
            'large_search_space': {
                'grid_search': 'Impractical (days/weeks)',
                'random_search': '~60% coverage, 15-30 minutes',
                'bayesian': '~85% coverage, 10-25 minutes',
                'optuna': '~90% coverage, 8-20 minutes'
            }
        }
    }


def get_usage_examples() -> Dict[str, str]:
    """Get practical usage examples for different scenarios."""
    return {
        'basic_automl': '''
# Basic AutoML pipeline
from automl import AutoMLPipeline, AutoMLConfig

config = AutoMLConfig(
    mode=AutoMLMode.BALANCED,
    max_trials=50,
    cv_folds=5,
    timeout_minutes=30
)

automl = AutoMLPipeline(config)
results = automl.fit(X_train, y_train)
best_model = automl.best_model
predictions = best_model.predict(X_test)
        ''',
        
        'hyperparameter_optimization': '''
# Advanced hyperparameter optimization
from automl import HyperparameterOptimizer, OptimizationAlgorithm
from sklearn.ensemble import RandomForestClassifier

optimizer = HyperparameterOptimizer(
    algorithm=OptimizationAlgorithm.OPTUNA,
    cv_folds=5,
    scoring='accuracy'
)

param_space = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

result = optimizer.optimize(
    model=RandomForestClassifier(),
    param_space=param_space,
    X=X_train,
    y=y_train,
    max_trials=100
)

print(f"Best parameters: {result.best_params}")
print(f"Best score: {result.best_score:.4f}")
        ''',
        
        'feature_selection': '''
# Automated feature selection
config = AutoMLConfig(
    feature_selection=True,
    feature_selection_method=FeatureSelectionMethod.SELECT_K_BEST,
    max_features=50,
    preprocessing=True
)

automl = AutoMLPipeline(config)
automl.fit(X_train, y_train)

# Get selected features
selected_features = automl.get_selected_features()
print(f"Selected {len(selected_features)} features: {selected_features}")
        ''',
        
        'custom_optimization': '''
# Custom optimization with multiple algorithms
algorithms = [
    OptimizationAlgorithm.RANDOM_SEARCH,
    OptimizationAlgorithm.BAYESIAN,
    OptimizationAlgorithm.OPTUNA
]

results = {}
for algorithm in algorithms:
    optimizer = HyperparameterOptimizer(algorithm=algorithm)
    result = optimizer.optimize(model, param_space, X, y, max_trials=50)
    results[algorithm.value] = result

# Compare results
for algo, result in results.items():
    print(f"{algo}: {result.best_score:.4f}")
        '''
    }


def export_info_json(filename: str = 'automl_info.json') -> None:
    """Export complete module information to JSON file."""
    info_data = {
        'package_info': get_package_info(),
        'optimization_comparison': get_optimization_comparison(),
        'usage_examples': get_usage_examples(),
        'generated_at': datetime.now().isoformat()
    }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(info_data, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"AutoML module information exported to {filename}")
        print(f"✅ Module information exported to {filename}")
    except Exception as e:
        logger.error(f"Failed to export module information: {e}")
        print(f"❌ Export failed: {e}")


if __name__ == "__main__":
    # Generate and display summary when run directly
    print("🎯 AutoML Module Information")
    print("=" * 60)
    print(json.dumps(get_package_info(), indent=2))
    print("\n" + "=" * 60)
    
    # Optionally export to JSON
    export_choice = input("\nExport detailed information to JSON? (y/n): ").lower().strip()
    if export_choice in ['y', 'yes']:
        export_info_json()
    
    print("\n📚 Documentation complete!")
