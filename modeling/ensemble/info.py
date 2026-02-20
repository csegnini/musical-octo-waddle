"""
Ensemble Methods Module Information Module.

This module provides comprehensive information about the ensemble methods module
capabilities, features, and usage guidelines for advanced ensemble learning techniques.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


def get_package_info() -> Dict[str, Any]:
    """
    Get comprehensive ensemble methods module information.
    
    Returns:
        Dictionary containing complete module details
    """
    return {
        'module_name': 'Advanced Ensemble Methods Framework',
        'version': '1.0.0',
        'description': 'Comprehensive ensemble learning framework with multiple algorithms, automatic model selection, and advanced combination strategies',
        'author': 'AI ML Platform Team',
        'license': 'MIT',
        'created_date': '2025-07-27',
        'last_updated': datetime.now().isoformat(),
        'core_components': {
            'ensemble_models': {
                'file': '__init__.py',
                'lines_of_code': 892,
                'description': 'Advanced ensemble models with multiple algorithms and automatic optimization',
                'key_classes': ['EnsembleModel', 'EnsembleConfig', 'EnsembleMethod', 'CombinationStrategy'],
                'features': [
                    '7+ ensemble algorithms (Random Forest, Gradient Boosting, AdaBoost, Voting, Bagging, Stacking, Blending)',
                    'Automatic base model selection and optimization',
                    'Advanced combination strategies (voting, averaging, stacking)',
                    'Multi-level ensemble architectures',
                    'Cross-validation and out-of-fold predictions',
                    'Feature importance aggregation',
                    'Parallel training and prediction',
                    'Comprehensive ensemble evaluation and analysis'
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
        'supported_ensemble_methods': {
            'bagging_methods': {
                'random_forest': {
                    'description': 'Bootstrap aggregating with decision trees',
                    'algorithm_type': 'Bagging with random feature selection',
                    'use_cases': ['Classification', 'Regression', 'Feature importance', 'High-dimensional data'],
                    'advantages': ['Reduces overfitting', 'Handles missing values', 'Feature importance', 'Parallel training'],
                    'disadvantages': ['Can overfit with noisy data', 'Less interpretable than single tree'],
                    'key_parameters': ['n_estimators', 'max_features', 'max_depth', 'min_samples_split']
                },
                'extra_trees': {
                    'description': 'Extremely randomized trees with random thresholds',
                    'algorithm_type': 'Random forest variant with additional randomness',
                    'use_cases': ['High-dimensional data', 'Feature selection', 'Fast training'],
                    'advantages': ['Faster training', 'More randomness reduces overfitting', 'Good feature importance'],
                    'disadvantages': ['May have higher bias', 'Less accurate than Random Forest'],
                    'key_parameters': ['n_estimators', 'max_features', 'random_state']
                },
                'bagging_classifier': {
                    'description': 'Generic bagging with any base classifier',
                    'algorithm_type': 'Bootstrap aggregating with custom base estimators',
                    'use_cases': ['Reducing overfitting', 'Improving stability', 'Parallel training'],
                    'advantages': ['Works with any base model', 'Reduces variance', 'Parallel processing'],
                    'disadvantages': ['May not improve bias', 'Increased computational cost'],
                    'key_parameters': ['base_estimator', 'n_estimators', 'max_samples']
                }
            },
            'boosting_methods': {
                'gradient_boosting': {
                    'description': 'Sequential ensemble minimizing residual errors',
                    'algorithm_type': 'Gradient descent in function space',
                    'use_cases': ['High-accuracy predictions', 'Structured data', 'Competition tasks'],
                    'advantages': ['Excellent predictive performance', 'Handles different data types', 'Feature importance'],
                    'disadvantages': ['Prone to overfitting', 'Sequential training', 'Sensitive to outliers'],
                    'key_parameters': ['n_estimators', 'learning_rate', 'max_depth', 'subsample']
                },
                'adaptive_boosting': {
                    'description': 'AdaBoost - adaptive weight adjustment for misclassified samples',
                    'algorithm_type': 'Adaptive boosting with weight resampling',
                    'use_cases': ['Binary classification', 'Weak learner combination', 'Imbalanced data'],
                    'advantages': ['Simple algorithm', 'Good with weak learners', 'Theoretical guarantees'],
                    'disadvantages': ['Sensitive to noise', 'Can overfit', 'Binary classification focus'],
                    'key_parameters': ['n_estimators', 'learning_rate', 'base_estimator']
                },
                'extreme_gradient_boosting': {
                    'description': 'XGBoost - optimized gradient boosting implementation',
                    'algorithm_type': 'Enhanced gradient boosting with regularization',
                    'use_cases': ['Competition tasks', 'Large datasets', 'High-performance requirements'],
                    'advantages': ['State-of-the-art performance', 'Built-in regularization', 'Parallel processing'],
                    'disadvantages': ['Complex hyperparameter tuning', 'Memory intensive', 'Black box'],
                    'key_parameters': ['n_estimators', 'learning_rate', 'max_depth', 'reg_alpha', 'reg_lambda']
                },
                'light_gradient_boosting': {
                    'description': 'LightGBM - fast gradient boosting with leaf-wise growth',
                    'algorithm_type': 'Leaf-wise tree growth gradient boosting',
                    'use_cases': ['Large datasets', 'Fast training', 'Memory-efficient modeling'],
                    'advantages': ['Very fast training', 'Memory efficient', 'High accuracy'],
                    'disadvantages': ['Can overfit small datasets', 'Sensitive to parameters'],
                    'key_parameters': ['n_estimators', 'learning_rate', 'num_leaves', 'feature_fraction']
                }
            },
            'voting_methods': {
                'hard_voting': {
                    'description': 'Majority vote from multiple classifiers',
                    'algorithm_type': 'Democracy-based prediction combination',
                    'use_cases': ['Classification tasks', 'Diverse model combination', 'Robust predictions'],
                    'advantages': ['Simple and interpretable', 'Reduces overfitting', 'Works with any classifier'],
                    'disadvantages': ['Ignores prediction confidence', 'May not improve weak models'],
                    'combination_strategy': 'Majority class vote'
                },
                'soft_voting': {
                    'description': 'Probability-weighted average from multiple classifiers',
                    'algorithm_type': 'Weighted probability combination',
                    'use_cases': ['Classification with probability estimates', 'Confidence-aware predictions'],
                    'advantages': ['Uses prediction confidence', 'Smoother decision boundaries', 'Better calibration'],
                    'disadvantages': ['Requires probability estimates', 'More complex than hard voting'],
                    'combination_strategy': 'Weighted probability average'
                },
                'weighted_voting': {
                    'description': 'Voting with different weights for each model',
                    'algorithm_type': 'Performance-weighted model combination',
                    'use_cases': ['When models have different performance levels', 'Expert system combination'],
                    'advantages': ['Emphasizes better models', 'Flexible weighting schemes'],
                    'disadvantages': ['Requires weight optimization', 'Risk of overfitting to weights'],
                    'combination_strategy': 'Performance-based weighted combination'
                }
            },
            'stacking_methods': {
                'stacked_generalization': {
                    'description': 'Meta-learner trained on base model predictions',
                    'algorithm_type': 'Two-level learning with meta-model',
                    'use_cases': ['Maximum performance', 'Complex pattern learning', 'Competition tasks'],
                    'advantages': ['Can learn optimal combination', 'Flexible meta-learner choice', 'High performance'],
                    'disadvantages': ['Complex implementation', 'Risk of overfitting', 'Computational overhead'],
                    'meta_learners': ['Linear Regression', 'Logistic Regression', 'Neural Networks', 'Tree-based models']
                },
                'blending': {
                    'description': 'Simplified stacking with holdout validation',
                    'algorithm_type': 'Single holdout set for meta-model training',
                    'use_cases': ['Simpler alternative to stacking', 'Fast ensemble creation'],
                    'advantages': ['Simpler than stacking', 'Faster training', 'Less overfitting risk'],
                    'disadvantages': ['Uses less data', 'May be less optimal than stacking'],
                    'validation_strategy': 'Single holdout set for meta-learning'
                },
                'multi_level_stacking': {
                    'description': 'Multiple levels of stacking for complex ensembles',
                    'algorithm_type': 'Hierarchical ensemble with multiple meta-levels',
                    'use_cases': ['Complex problems', 'Maximum performance requirements', 'Research applications'],
                    'advantages': ['Very high performance potential', 'Flexible architecture'],
                    'disadvantages': ['Very complex', 'High overfitting risk', 'Computational intensive'],
                    'architecture': 'Level 0 (base) → Level 1 (meta) → Level 2 (meta-meta) → ...'
                }
            }
        },
        'combination_strategies': {
            'averaging_methods': {
                'simple_averaging': {
                    'description': 'Equal weight average of all model predictions',
                    'formula': '(p1 + p2 + ... + pn) / n',
                    'use_cases': 'When all models have similar performance',
                    'advantages': ['Simple', 'Robust', 'No hyperparameters'],
                    'disadvantages': ['Ignores model quality differences']
                },
                'weighted_averaging': {
                    'description': 'Performance-weighted average of model predictions',
                    'formula': 'Σ(wi * pi) where wi are weights',
                    'use_cases': 'When models have different performance levels',
                    'advantages': ['Emphasizes better models', 'Flexible'],
                    'disadvantages': ['Requires weight optimization']
                },
                'rank_averaging': {
                    'description': 'Average of prediction ranks instead of values',
                    'use_cases': 'Ranking and recommendation problems',
                    'advantages': ['Scale-invariant', 'Robust to outliers'],
                    'disadvantages': ['Loses magnitude information']
                }
            },
            'selection_methods': {
                'dynamic_selection': {
                    'description': 'Select best model for each prediction instance',
                    'selection_criteria': ['Local accuracy', 'Competence measures', 'Distance-based'],
                    'advantages': ['Adaptive to instance characteristics', 'Can achieve high performance'],
                    'disadvantages': ['Complex implementation', 'Computational overhead']
                },
                'clustering_selection': {
                    'description': 'Select models based on input space clustering',
                    'selection_strategy': 'Different models for different regions',
                    'advantages': ['Specialized models for different patterns'],
                    'disadvantages': ['Requires clustering overhead']
                }
            },
            'optimization_methods': {
                'evolutionary_combination': {
                    'description': 'Genetic algorithms for optimal model combination',
                    'optimization_target': 'Model weights and selection',
                    'advantages': ['Global optimization', 'Handles complex objective functions'],
                    'disadvantages': ['Computationally expensive', 'No convergence guarantees']
                },
                'bayesian_combination': {
                    'description': 'Bayesian model averaging with uncertainty quantification',
                    'theoretical_foundation': 'Bayesian model averaging principles',
                    'advantages': ['Principled uncertainty', 'Theoretical foundation'],
                    'disadvantages': ['Complex implementation', 'Computational requirements']
                }
            }
        },
        'advanced_features': {
            'automatic_ensemble_construction': {
                'description': 'Automatic selection and combination of base models',
                'model_selection_criteria': [
                    'Cross-validation performance',
                    'Model diversity measures',
                    'Computational efficiency',
                    'Complementary strengths'
                ],
                'diversity_measures': [
                    'Disagreement measure',
                    'Double-fault measure',
                    'Correlation coefficient',
                    'Q-statistic',
                    'Entropy-based measures'
                ],
                'selection_algorithms': [
                    'Forward selection',
                    'Backward elimination',
                    'Genetic algorithms',
                    'Pareto optimization'
                ]
            },
            'ensemble_pruning': {
                'description': 'Removing redundant or harmful ensemble members',
                'pruning_criteria': [
                    'Individual model performance',
                    'Contribution to ensemble performance',
                    'Diversity contribution',
                    'Computational cost'
                ],
                'pruning_algorithms': [
                    'Ranking-based pruning',
                    'Clustering-based pruning',
                    'Optimization-based pruning',
                    'Margin-based pruning'
                ],
                'benefits': [
                    'Reduced computational cost',
                    'Improved generalization',
                    'Simplified deployment',
                    'Better interpretability'
                ]
            },
            'online_ensemble_learning': {
                'description': 'Adaptive ensemble learning for streaming data',
                'adaptation_mechanisms': [
                    'Model weight adjustment',
                    'Model addition/removal',
                    'Concept drift detection',
                    'Forgetting mechanisms'
                ],
                'drift_detection': [
                    'Statistical change detection',
                    'Performance degradation monitoring',
                    'Distribution shift detection',
                    'Ensemble diversity tracking'
                ],
                'use_cases': [
                    'Streaming data applications',
                    'Non-stationary environments',
                    'Real-time prediction systems',
                    'Adaptive recommender systems'
                ]
            },
            'multi_objective_ensembles': {
                'description': 'Ensembles optimizing multiple objectives simultaneously',
                'objective_types': [
                    'Accuracy vs. interpretability',
                    'Performance vs. computational cost',
                    'Bias vs. variance',
                    'Fairness vs. accuracy'
                ],
                'optimization_approaches': [
                    'Pareto frontier optimization',
                    'Weighted objective functions',
                    'Constraint-based optimization',
                    'Multi-criteria decision making'
                ]
            }
        },
        'performance_characteristics': {
            'accuracy_improvements': {
                'typical_gains': {
                    'bagging_methods': '2-5% accuracy improvement over single models',
                    'boosting_methods': '5-15% accuracy improvement with proper tuning',
                    'voting_ensembles': '1-3% improvement with diverse models',
                    'stacking_methods': '3-8% improvement with good meta-learner'
                },
                'factors_affecting_performance': [
                    'Base model diversity',
                    'Individual model quality',
                    'Problem complexity',
                    'Dataset size and quality',
                    'Ensemble combination strategy'
                ]
            },
            'computational_complexity': {
                'training_time': {
                    'bagging': 'Linear scaling with number of models (parallelizable)',
                    'boosting': 'Sequential training, no parallelization',
                    'voting': 'Sum of individual model training times',
                    'stacking': '2x time due to cross-validation requirements'
                },
                'prediction_time': {
                    'bagging': 'Linear scaling with ensemble size',
                    'boosting': 'Linear scaling with ensemble size',
                    'voting': 'Sum of individual model prediction times',
                    'stacking': 'Base models + meta-model prediction time'
                },
                'memory_requirements': {
                    'model_storage': 'Sum of individual model sizes',
                    'training_memory': 'Depends on parallelization strategy',
                    'optimization': 'Model compression and pruning techniques'
                }
            },
            'scalability_characteristics': {
                'data_size_scaling': {
                    'small_datasets': '< 10K samples - risk of overfitting with complex ensembles',
                    'medium_datasets': '10K-1M samples - good performance with most ensemble methods',
                    'large_datasets': '> 1M samples - all ensemble methods applicable, focus on efficiency'
                },
                'feature_scaling': {
                    'low_dimensional': '< 100 features - all methods work well',
                    'medium_dimensional': '100-10K features - tree-based ensembles preferred',
                    'high_dimensional': '> 10K features - feature selection and regularization important'
                }
            }
        },
        'problem_type_support': [
            {
                'category': 'Classification',
                'supported_types': [
                    {
                        'name': 'Binary Classification',
                        'ensemble_methods': ['All methods supported'],
                        'combination_strategies': ['Voting', 'Averaging', 'Stacking'],
                        'evaluation_metrics': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                        'typical_performance': '85-95% accuracy with good ensembles'
                    },
                    {
                        'name': 'Multi-class Classification',
                        'ensemble_methods': ['Random Forest', 'Gradient Boosting', 'Voting', 'Stacking'],
                        'combination_strategies': ['Soft voting preferred', 'Probability averaging'],
                        'evaluation_metrics': ['Accuracy', 'Macro/Micro F1', 'Confusion Matrix'],
                        'typical_performance': '80-92% accuracy depending on number of classes'
                    },
                    {
                        'name': 'Multi-label Classification',
                        'ensemble_methods': ['Binary relevance ensembles', 'Label powerset ensembles'],
                        'combination_strategies': ['Independent binary combination', 'Joint probability'],
                        'evaluation_metrics': ['Hamming loss', 'Jaccard score', 'Label ranking'],
                        'typical_performance': 'Problem-dependent, focus on label dependencies'
                    }
                ]
            },
            {
                'category': 'Regression',
                'supported_types': [
                    {
                        'name': 'Single-output Regression',
                        'ensemble_methods': ['Random Forest', 'Gradient Boosting', 'Voting Regressor'],
                        'combination_strategies': ['Averaging', 'Weighted averaging', 'Stacking'],
                        'evaluation_metrics': ['MAE', 'MSE', 'RMSE', 'R²'],
                        'typical_performance': 'R² > 0.8 for good ensembles'
                    },
                    {
                        'name': 'Multi-output Regression',
                        'ensemble_methods': ['Multi-output Random Forest', 'Multi-output Gradient Boosting'],
                        'combination_strategies': ['Independent output combination', 'Joint modeling'],
                        'evaluation_metrics': ['Multi-output MAE', 'Multi-output R²'],
                        'typical_performance': 'Varies by output correlation'
                    }
                ]
            },
            {
                'category': 'Ranking',
                'supported_types': [
                    {
                        'name': 'Learning to Rank',
                        'ensemble_methods': ['RankBoost', 'Ensemble ranking methods'],
                        'combination_strategies': ['Rank averaging', 'Score combination'],
                        'evaluation_metrics': ['NDCG', 'MAP', 'MRR'],
                        'typical_performance': 'Significant improvements over single rankers'
                    }
                ]
            }
        ],
        'integration_capabilities': {
            'scikit_learn_integration': {
                'estimator_compatibility': 'Full compliance with sklearn estimator interface',
                'pipeline_support': 'Works with sklearn pipelines and transformers',
                'cross_validation': 'Compatible with all sklearn CV strategies',
                'grid_search': 'Hyperparameter optimization support',
                'feature_selection': 'Integration with sklearn feature selection'
            },
            'external_library_support': {
                'xgboost_integration': 'Native XGBoost model support',
                'lightgbm_integration': 'Native LightGBM model support',
                'catboost_integration': 'CatBoost model support',
                'deep_learning_integration': 'Neural network ensemble members',
                'custom_models': 'Support for user-defined base models'
            },
            'parallel_processing': {
                'training_parallelization': 'Parallel training of ensemble members',
                'prediction_parallelization': 'Parallel prediction across models',
                'cross_validation_parallelization': 'Parallel CV fold processing',
                'optimization_parallelization': 'Parallel hyperparameter optimization'
            },
            'model_persistence': {
                'serialization_formats': ['Pickle', 'Joblib', 'Custom formats'],
                'version_compatibility': 'Backward compatibility maintenance',
                'compression': 'Model compression for storage efficiency',
                'deployment_formats': 'Production-ready model formats'
            }
        },
        'use_cases_applications': [
            {
                'domain': 'Competitive Machine Learning',
                'applications': ['Kaggle competitions', 'Data science contests', 'Benchmark datasets'],
                'recommended_methods': ['Stacking', 'Multi-level ensembles', 'XGBoost/LightGBM'],
                'key_strategies': 'Maximum performance, complex model combinations',
                'typical_improvements': '5-15% over single models'
            },
            {
                'domain': 'Production Machine Learning',
                'applications': ['Recommendation systems', 'Fraud detection', 'Risk assessment'],
                'recommended_methods': ['Random Forest', 'Gradient Boosting', 'Voting ensembles'],
                'key_strategies': 'Balance performance and interpretability, robustness',
                'typical_improvements': '2-8% with improved reliability'
            },
            {
                'domain': 'Healthcare & Medical',
                'applications': ['Disease diagnosis', 'Drug discovery', 'Treatment recommendation'],
                'recommended_methods': ['Random Forest', 'Bagging', 'Voting with confidence'],
                'key_strategies': 'High reliability, uncertainty quantification, interpretability',
                'regulatory_considerations': 'FDA approval requirements, explainability'
            },
            {
                'domain': 'Financial Services',
                'applications': ['Credit scoring', 'Algorithmic trading', 'Market prediction'],
                'recommended_methods': ['Gradient Boosting', 'Stacking', 'Online ensembles'],
                'key_strategies': 'High accuracy, real-time prediction, risk management',
                'regulatory_considerations': 'Model interpretability, fairness, stability'
            },
            {
                'domain': 'Manufacturing & Quality Control',
                'applications': ['Defect detection', 'Process optimization', 'Predictive maintenance'],
                'recommended_methods': ['Random Forest', 'Gradient Boosting', 'Online ensembles'],
                'key_strategies': 'Real-time processing, high reliability, cost optimization',
                'operational_requirements': 'Edge deployment, minimal latency'
            },
            {
                'domain': 'Natural Language Processing',
                'applications': ['Text classification', 'Sentiment analysis', 'Named entity recognition'],
                'recommended_methods': ['Voting classifiers', 'Stacking with diverse features'],
                'key_strategies': 'Combine different feature representations, handle text diversity',
                'technical_considerations': 'Feature engineering, dimensionality'
            }
        ],
        'best_practices': {
            'ensemble_design': {
                'diversity_maximization': [
                    'Use different algorithms (tree-based, linear, neural)',
                    'Different feature subsets or transformations',
                    'Different training data samples',
                    'Different hyperparameter settings',
                    'Different random seeds for stochastic algorithms'
                ],
                'base_model_selection': [
                    'Ensure individual models are better than random',
                    'Balance individual performance with diversity',
                    'Avoid including too many weak models',
                    'Consider computational constraints',
                    'Test model complementarity'
                ],
                'combination_strategy_selection': [
                    'Simple averaging for similar-performance models',
                    'Weighted averaging when models have different quality',
                    'Voting for classification with diverse models',
                    'Stacking for maximum performance with risk of overfitting'
                ]
            },
            'training_strategies': {
                'cross_validation_design': [
                    'Use appropriate CV strategy for ensemble type',
                    'Stratified CV for classification',
                    'Time series CV for temporal data',
                    'Group CV for grouped data structures'
                ],
                'overfitting_prevention': [
                    'Use proper validation methodology',
                    'Limit ensemble complexity appropriately',
                    'Apply regularization in meta-learners',
                    'Monitor validation performance carefully'
                ],
                'computational_optimization': [
                    'Parallelize training when possible',
                    'Use early stopping for boosting methods',
                    'Consider ensemble pruning for deployment',
                    'Cache intermediate results'
                ]
            },
            'evaluation_validation': {
                'performance_assessment': [
                    'Use multiple evaluation metrics',
                    'Test on truly held-out data',
                    'Compare against strong baselines',
                    'Analyze individual model contributions'
                ],
                'robustness_testing': [
                    'Test performance across different data distributions',
                    'Evaluate sensitivity to hyperparameters',
                    'Check performance degradation with fewer models',
                    'Test computational performance requirements'
                ]
            },
            'deployment_considerations': {
                'production_readiness': [
                    'Optimize prediction latency',
                    'Minimize memory requirements',
                    'Ensure model serialization compatibility',
                    'Plan for model updates and versioning'
                ],
                'monitoring_maintenance': [
                    'Monitor individual model performance',
                    'Track ensemble performance metrics',
                    'Detect and handle concept drift',
                    'Plan for model retraining schedules'
                ]
            }
        }
    }


def get_ensemble_method_comparison() -> Dict[str, Dict[str, Any]]:
    """Get detailed comparison of ensemble methods."""
    return {
        'random_forest': {
            'algorithm_family': 'Bagging',
            'base_learner': 'Decision Trees',
            'training_paradigm': 'Parallel',
            'strengths': [
                'Excellent out-of-box performance',
                'Handles mixed data types well',
                'Built-in feature importance',
                'Robust to overfitting',
                'Handles missing values',
                'No need for feature scaling'
            ],
            'weaknesses': [
                'Can overfit with very noisy data',
                'Memory intensive for large ensembles',
                'Less effective on linear relationships',
                'Biased toward categorical variables with more levels'
            ],
            'best_for': [
                'Tabular data with mixed features',
                'When interpretability is important',
                'Baseline ensemble method',
                'High-dimensional data',
                'When training time is limited'
            ],
            'hyperparameters': {
                'n_estimators': '100-1000 (more is usually better)',
                'max_features': 'sqrt(n) for classification, n/3 for regression',
                'max_depth': 'None (fully grown) or tune for regularization',
                'min_samples_split': '2-10 depending on dataset size'
            },
            'computational_complexity': {
                'training': 'O(n * log(n) * m * k) where n=samples, m=features, k=trees',
                'prediction': 'O(log(depth) * k)',
                'memory': 'O(k * tree_size)'
            },
            'typical_performance': {
                'accuracy_gain': '2-5% over single decision tree',
                'training_time': 'Fast with parallelization',
                'prediction_time': 'Fast',
                'interpretability': 'Medium (feature importance available)'
            }
        },
        'gradient_boosting': {
            'algorithm_family': 'Boosting',
            'base_learner': 'Weak learners (usually trees)',
            'training_paradigm': 'Sequential',
            'strengths': [
                'Excellent predictive performance',
                'Handles different data types',
                'Feature importance available',
                'Good for competitions',
                'Flexible loss functions',
                'Can handle missing values (in modern implementations)'
            ],
            'weaknesses': [
                'Prone to overfitting',
                'Sensitive to hyperparameters',
                'Sequential training (harder to parallelize)',
                'Sensitive to outliers',
                'Requires careful tuning'
            ],
            'best_for': [
                'Maximum predictive performance',
                'Structured/tabular data',
                'Competition settings',
                'When training time is not critical',
                'Non-linear relationships'
            ],
            'hyperparameters': {
                'n_estimators': '100-1000 (with early stopping)',
                'learning_rate': '0.01-0.3 (lower is usually better)',
                'max_depth': '3-8 (shallow trees preferred)',
                'subsample': '0.8-1.0 for stochastic gradient boosting'
            },
            'computational_complexity': {
                'training': 'O(iterations * n * log(n) * m)',
                'prediction': 'O(iterations * log(depth))',
                'memory': 'O(iterations * tree_size)'
            },
            'typical_performance': {
                'accuracy_gain': '5-15% over single models with tuning',
                'training_time': 'Moderate to slow',
                'prediction_time': 'Fast',
                'interpretability': 'Medium (feature importance, partial dependence)'
            }
        },
        'xgboost': {
            'algorithm_family': 'Advanced Boosting',
            'base_learner': 'Gradient boosted trees',
            'training_paradigm': 'Sequential with optimizations',
            'strengths': [
                'State-of-the-art performance on structured data',
                'Built-in regularization',
                'Parallel processing capabilities',
                'Handles missing values automatically',
                'Cross-validation built-in',
                'Extensive hyperparameter options'
            ],
            'weaknesses': [
                'Complex hyperparameter tuning',
                'Can be memory intensive',
                'Prone to overfitting without proper tuning',
                'Black box nature',
                'Requires domain expertise for optimal performance'
            ],
            'best_for': [
                'Kaggle competitions',
                'Structured data with high performance requirements',
                'Large datasets',
                'When computational resources are available',
                'Complex non-linear patterns'
            ],
            'hyperparameters': {
                'n_estimators': '100-10000 with early stopping',
                'learning_rate': '0.01-0.3',
                'max_depth': '3-10',
                'reg_alpha': '0-10 (L1 regularization)',
                'reg_lambda': '0-10 (L2 regularization)',
                'subsample': '0.6-1.0'
            },
            'computational_complexity': {
                'training': 'Optimized with parallelization',
                'prediction': 'Very fast',
                'memory': 'Optimized memory usage'
            },
            'typical_performance': {
                'accuracy_gain': '8-20% over basic methods',
                'training_time': 'Moderate with parallelization',
                'prediction_time': 'Very fast',
                'interpretability': 'Low to medium (SHAP values available)'
            }
        },
        'voting_classifier': {
            'algorithm_family': 'Meta-ensemble',
            'base_learner': 'Any combination of classifiers',
            'training_paradigm': 'Parallel base model training',
            'strengths': [
                'Simple and robust',
                'Works with any base classifiers',
                'Reduces overfitting through averaging',
                'Easy to implement and understand',
                'Good baseline ensemble method'
            ],
            'weaknesses': [
                'Performance limited by base model quality',
                'Equal weighting may not be optimal',
                'Requires base models to be diverse',
                'May not improve much over best base model'
            ],
            'best_for': [
                'Combining diverse model types',
                'When simplicity is valued',
                'Baseline ensemble approach',
                'When base models have similar performance',
                'Quick ensemble implementation'
            ],
            'combination_strategies': {
                'hard_voting': 'Majority class vote',
                'soft_voting': 'Average of predicted probabilities',
                'weighted_voting': 'Performance-weighted combination'
            },
            'computational_complexity': {
                'training': 'Sum of base model training times',
                'prediction': 'Sum of base model prediction times',
                'memory': 'Sum of base model memory requirements'
            },
            'typical_performance': {
                'accuracy_gain': '1-3% over best individual model',
                'training_time': 'Depends on base models',
                'prediction_time': 'Sum of base model times',
                'interpretability': 'Depends on base models'
            }
        },
        'stacking': {
            'algorithm_family': 'Meta-learning',
            'base_learner': 'Any combination + meta-learner',
            'training_paradigm': 'Two-level learning',
            'strengths': [
                'Can achieve very high performance',
                'Learns optimal combination automatically',
                'Flexible meta-learner choice',
                'Can capture complex interactions',
                'Theoretically well-founded'
            ],
            'weaknesses': [
                'Complex implementation',
                'Risk of overfitting',
                'Computationally expensive',
                'Requires careful validation',
                'Less interpretable'
            ],
            'best_for': [
                'Maximum performance requirements',
                'Competition settings',
                'When computational resources are available',
                'Complex pattern recognition',
                'Research applications'
            ],
            'meta_learners': {
                'linear_regression': 'Simple, interpretable, fast',
                'logistic_regression': 'Good for classification, interpretable',
                'neural_network': 'Can learn complex combinations',
                'tree_models': 'Non-linear combinations, feature importance'
            },
            'computational_complexity': {
                'training': '2x base model training + meta-model training',
                'prediction': 'Base models + meta-model prediction',
                'memory': 'Base models + meta-model + validation predictions'
            },
            'typical_performance': {
                'accuracy_gain': '3-8% over voting methods',
                'training_time': 'Slow (cross-validation required)',
                'prediction_time': 'Moderate',
                'interpretability': 'Low (depends on meta-learner)'
            }
        }
    }


def get_implementation_examples() -> Dict[str, str]:
    """Get comprehensive implementation examples."""
    return {
        'basic_random_forest': '''
# Basic Random Forest Ensemble Example
from modeling.ensemble import EnsembleModel, EnsembleConfig, EnsembleMethod

# Configure Random Forest
config = EnsembleConfig(
    method=EnsembleMethod.RANDOM_FOREST,
    n_estimators=100,
    max_features='sqrt',
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1  # Use all available cores
)

# Create and train the ensemble
ensemble = EnsembleModel(config, name="Random_Forest_Classifier")
ensemble.fit(X_train, y_train)

# Make predictions
predictions = ensemble.predict(X_test)
probabilities = ensemble.predict_proba(X_test)

# Evaluate performance
metrics = ensemble.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"F1-Score: {metrics['f1_score']:.3f}")

# Analyze feature importance
feature_importance = ensemble.get_feature_importance()
ensemble.plot_feature_importance(top_n=20)
''',

        'gradient_boosting_example': '''
# Gradient Boosting Ensemble Example
from modeling.ensemble import EnsembleModel, EnsembleConfig, EnsembleMethod

# Configure Gradient Boosting with early stopping
config = EnsembleConfig(
    method=EnsembleMethod.GRADIENT_BOOSTING,
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=10
)

# Train with early stopping
ensemble = EnsembleModel(config, name="GBM_Classifier")
ensemble.fit(X_train, y_train)

# Get optimal number of estimators
print(f"Optimal n_estimators: {ensemble.best_n_estimators_}")

# Make predictions
predictions = ensemble.predict(X_test)

# Plot training progress
ensemble.plot_training_progress()

# Evaluate performance
metrics = ensemble.evaluate(X_test, y_test)
print(f"Test Accuracy: {metrics['accuracy']:.3f}")
''',

        'voting_ensemble_example': '''
# Voting Ensemble Example
from modeling.ensemble import EnsembleModel, EnsembleConfig, EnsembleMethod
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Define base estimators
base_estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
    ('lr', LogisticRegression(random_state=42))
]

# Configure voting ensemble
config = EnsembleConfig(
    method=EnsembleMethod.VOTING,
    base_estimators=base_estimators,
    voting='soft',  # Use probability-based voting
    weights=[2, 1, 1]  # Give Random Forest higher weight
)

# Train voting ensemble
ensemble = EnsembleModel(config, name="Voting_Classifier")
ensemble.fit(X_train, y_train)

# Make predictions
predictions = ensemble.predict(X_test)
probabilities = ensemble.predict_proba(X_test)

# Analyze individual model contributions
individual_predictions = ensemble.get_individual_predictions(X_test)
for name, pred in individual_predictions.items():
    individual_accuracy = accuracy_score(y_test, pred)
    print(f"{name} accuracy: {individual_accuracy:.3f}")

# Evaluate ensemble
metrics = ensemble.evaluate(X_test, y_test)
print(f"Ensemble accuracy: {metrics['accuracy']:.3f}")
''',

        'stacking_ensemble_example': '''
# Stacking Ensemble Example
from modeling.ensemble import EnsembleModel, EnsembleConfig, EnsembleMethod
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Define base (level-0) models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, random_state=42))
]

# Define meta (level-1) model
meta_model = LogisticRegression(random_state=42)

# Configure stacking ensemble
config = EnsembleConfig(
    method=EnsembleMethod.STACKING,
    base_estimators=base_models,
    meta_estimator=meta_model,
    cv=5,  # 5-fold cross-validation for meta-features
    use_probas=True,  # Use prediction probabilities as meta-features
    use_features_in_secondary=False  # Don't include original features in meta-model
)

# Train stacking ensemble
ensemble = EnsembleModel(config, name="Stacking_Classifier")
ensemble.fit(X_train, y_train)

# Make predictions
predictions = ensemble.predict(X_test)

# Analyze meta-features
meta_features = ensemble.get_meta_features(X_test)
print(f"Meta-features shape: {meta_features.shape}")

# Evaluate performance
metrics = ensemble.evaluate(X_test, y_test)
print(f"Stacking accuracy: {metrics['accuracy']:.3f}")

# Compare with individual base models
base_scores = ensemble.get_base_model_scores(X_test, y_test)
print("Base model performances:")
for name, score in base_scores.items():
    print(f"{name}: {score:.3f}")
''',

        'automatic_ensemble_selection': '''
# Automatic Ensemble Selection Example
from modeling.ensemble import EnsembleModel, EnsembleConfig, EnsembleMethod

# Configure automatic ensemble selection
config = EnsembleConfig(
    method=EnsembleMethod.AUTO_ENSEMBLE,
    candidate_methods=[
        'random_forest', 'gradient_boosting', 'extra_trees',
        'xgboost', 'lightgbm', 'voting', 'stacking'
    ],
    selection_metric='f1_score',
    cv_folds=5,
    diversity_weight=0.3,  # Balance performance and diversity
    max_ensemble_size=5,   # Maximum number of models in final ensemble
    optimization_budget=3600  # 1 hour optimization budget
)

# Automatic ensemble construction
ensemble = EnsembleModel(config, name="Auto_Ensemble")
ensemble.fit(X_train, y_train)

# Get selected ensemble composition
selected_models = ensemble.get_selected_models()
print("Selected ensemble composition:")
for model_name, weight in selected_models.items():
    print(f"{model_name}: weight = {weight:.3f}")

# Analyze selection process
selection_history = ensemble.get_selection_history()
ensemble.plot_selection_process()

# Make predictions
predictions = ensemble.predict(X_test)

# Comprehensive evaluation
metrics = ensemble.evaluate(X_test, y_test)
print(f"Auto-ensemble performance: {metrics}")
''',

        'ensemble_with_feature_selection': '''
# Ensemble with Feature Selection Example
from modeling.ensemble import EnsembleModel, EnsembleConfig, EnsembleMethod
from sklearn.feature_selection import SelectKBest, f_classif

# Configure ensemble with feature selection
config = EnsembleConfig(
    method=EnsembleMethod.RANDOM_FOREST,
    n_estimators=100,
    feature_selection=SelectKBest(score_func=f_classif, k=50),
    feature_selection_cv=True,  # Use CV to select optimal k
    preprocessing_pipeline=True,  # Include preprocessing
    scale_features=True
)

# Train ensemble with automatic feature selection
ensemble = EnsembleModel(config, name="RF_with_Feature_Selection")
ensemble.fit(X_train, y_train)

# Get selected features
selected_features = ensemble.get_selected_features()
print(f"Selected {len(selected_features)} features out of {X_train.shape[1]}")

# Analyze feature importance on selected features
feature_importance = ensemble.get_feature_importance()
ensemble.plot_feature_importance(feature_names=selected_features)

# Make predictions
predictions = ensemble.predict(X_test)

# Compare performance with and without feature selection
metrics_with_selection = ensemble.evaluate(X_test, y_test)

# Train baseline without feature selection
baseline_config = EnsembleConfig(
    method=EnsembleMethod.RANDOM_FOREST,
    n_estimators=100
)
baseline = EnsembleModel(baseline_config, name="RF_baseline")
baseline.fit(X_train, y_train)
metrics_baseline = baseline.evaluate(X_test, y_test)

print(f"With feature selection: {metrics_with_selection['f1_score']:.3f}")
print(f"Without feature selection: {metrics_baseline['f1_score']:.3f}")
'''
    }


def get_optimization_guidelines() -> Dict[str, Any]:
    """Get optimization and tuning guidelines."""
    return {
        'hyperparameter_optimization': {
            'random_forest_tuning': {
                'primary_parameters': {
                    'n_estimators': {
                        'range': '50-1000',
                        'default': '100',
                        'tuning_tip': 'More is usually better, but diminishing returns after 200-500',
                        'computational_impact': 'Linear increase in training and prediction time'
                    },
                    'max_features': {
                        'range': 'sqrt, log2, 0.3-0.8, None',
                        'default': 'sqrt for classification, 1/3 for regression',
                        'tuning_tip': 'sqrt is usually good, try log2 for high-dimensional data',
                        'impact': 'Controls randomness and overfitting'
                    },
                    'max_depth': {
                        'range': 'None, 5-50',
                        'default': 'None (fully grown)',
                        'tuning_tip': 'Limit for regularization, especially with small datasets',
                        'impact': 'Controls model complexity and overfitting'
                    }
                },
                'secondary_parameters': {
                    'min_samples_split': '2-20',
                    'min_samples_leaf': '1-10',
                    'bootstrap': 'True (default), False for extra trees behavior'
                }
            },
            'gradient_boosting_tuning': {
                'primary_parameters': {
                    'n_estimators': {
                        'range': '100-2000',
                        'default': '100',
                        'tuning_tip': 'Use early stopping, start high and let it stop early',
                        'early_stopping': 'Monitor validation score to prevent overfitting'
                    },
                    'learning_rate': {
                        'range': '0.01-0.3',
                        'default': '0.1',
                        'tuning_tip': 'Lower rates often perform better but need more estimators',
                        'relationship': 'Inverse relationship with n_estimators'
                    },
                    'max_depth': {
                        'range': '2-10',
                        'default': '3',
                        'tuning_tip': 'Shallow trees (3-6) usually work best',
                        'impact': 'Deeper trees increase overfitting risk'
                    }
                },
                'regularization_parameters': {
                    'subsample': '0.5-1.0 (stochastic gradient boosting)',
                    'min_samples_split': '2-20',
                    'min_samples_leaf': '1-10',
                    'max_features': 'None, sqrt, log2'
                }
            },
            'xgboost_tuning': {
                'step_by_step_approach': {
                    'step_1_basic': {
                        'parameters': ['n_estimators', 'learning_rate'],
                        'strategy': 'Fix other parameters, tune these first',
                        'typical_values': 'n_estimators=1000, learning_rate=0.1'
                    },
                    'step_2_tree_structure': {
                        'parameters': ['max_depth', 'min_child_weight'],
                        'strategy': 'Grid search these together',
                        'typical_ranges': 'max_depth: 3-10, min_child_weight: 1-6'
                    },
                    'step_3_regularization': {
                        'parameters': ['reg_alpha', 'reg_lambda'],
                        'strategy': 'Add regularization to prevent overfitting',
                        'typical_ranges': 'Both 0-10, start with 0 and increase if overfitting'
                    },
                    'step_4_sampling': {
                        'parameters': ['subsample', 'colsample_bytree'],
                        'strategy': 'Fine-tune sampling parameters',
                        'typical_ranges': 'Both 0.6-1.0'
                    }
                }
            },
            'voting_ensemble_tuning': {
                'model_selection': {
                    'diversity_principle': 'Select models with different strengths/weaknesses',
                    'performance_threshold': 'Only include models better than random baseline',
                    'recommended_combinations': [
                        'Tree-based + Linear + SVM',
                        'Random Forest + Gradient Boosting + Neural Network',
                        'Multiple algorithms with different hyperparameters'
                    ]
                },
                'weight_optimization': {
                    'equal_weights': 'Good starting point, often works well',
                    'performance_based': 'Weight by cross-validation performance',
                    'grid_search': 'Search over weight combinations',
                    'optimization_algorithms': 'Use scipy.optimize for optimal weights'
                }
            },
            'stacking_optimization': {
                'base_model_selection': {
                    'diversity_requirement': 'Ensure diverse base models',
                    'performance_requirement': 'All base models should be reasonably good',
                    'computational_consideration': 'Balance performance with training time'
                },
                'meta_model_selection': {
                    'simple_meta_learners': 'Linear/Logistic Regression often work well',
                    'complex_meta_learners': 'Neural networks, tree models for complex patterns',
                    'regularization': 'Important to prevent overfitting in meta-learner'
                },
                'cross_validation_strategy': {
                    'fold_selection': '5-10 folds typical, more for small datasets',
                    'stratification': 'Use stratified CV for classification',
                    'time_series': 'Use time series CV for temporal data'
                }
            }
        },
        'performance_optimization': {
            'computational_efficiency': {
                'parallel_processing': {
                    'training_parallelization': 'Use n_jobs=-1 for tree-based methods',
                    'prediction_parallelization': 'Parallelize ensemble member predictions',
                    'memory_considerations': 'Balance parallelization with memory usage'
                },
                'early_stopping': {
                    'boosting_methods': 'Essential for gradient boosting methods',
                    'monitoring_metric': 'Use validation set to monitor performance',
                    'patience_setting': '10-50 iterations depending on dataset size'
                },
                'model_compression': {
                    'ensemble_pruning': 'Remove least contributing models',
                    'feature_selection': 'Reduce input dimensionality',
                    'quantization': 'Reduce model precision for deployment'
                }
            },
            'memory_optimization': {
                'batch_processing': 'Process large datasets in batches',
                'feature_selection': 'Reduce memory footprint with fewer features',
                'model_serialization': 'Use efficient serialization formats',
                'garbage_collection': 'Explicit memory management for large ensembles'
            }
        },
        'evaluation_strategies': {
            'cross_validation_best_practices': {
                'cv_strategy_selection': {
                    'standard_cv': 'K-fold for general problems',
                    'stratified_cv': 'Maintains class distribution',
                    'time_series_cv': 'Respects temporal order',
                    'group_cv': 'Prevents data leakage in grouped data'
                },
                'fold_number_selection': {
                    'small_datasets': '10-fold or leave-one-out',
                    'medium_datasets': '5-10 fold',
                    'large_datasets': '3-5 fold',
                    'computational_constraints': 'Fewer folds for expensive models'
                }
            },
            'metric_selection': {
                'classification_metrics': {
                    'balanced_datasets': 'Accuracy, F1-score',
                    'imbalanced_datasets': 'Precision, Recall, ROC-AUC, PR-AUC',
                    'multi_class': 'Macro/Micro F1, per-class metrics',
                    'business_metrics': 'Custom metrics aligned with business goals'
                },
                'regression_metrics': {
                    'general_purpose': 'RMSE, MAE',
                    'scale_independent': 'MAPE, R²',
                    'outlier_robust': 'MAE, Huber loss',
                    'business_metrics': 'Domain-specific error measures'
                }
            }
        }
    }


def generate_info_summary() -> str:
    """Generate a comprehensive summary of the ensemble methods module."""
    info = get_package_info()
    methods = get_ensemble_method_comparison()
    
    summary = f"""
# Ensemble Methods Module Summary

## Overview
{info['description']}

**Version:** {info['version']}
**Last Updated:** {info['last_updated']}

## Key Capabilities
- **{len(methods)} Core Ensemble Methods** covering all major ensemble paradigms
- **Automatic Ensemble Construction** with intelligent model selection
- **Advanced Combination Strategies** (voting, averaging, stacking, blending)
- **Multi-level Ensemble Architectures** for maximum performance
- **Comprehensive Evaluation and Analysis** tools
- **Production-Ready Optimization** with parallel processing

## Supported Ensemble Methods
### Bagging Methods
- **Random Forest:** Bootstrap aggregating with decision trees
- **Extra Trees:** Extremely randomized trees
- **Bagging:** Generic bootstrap aggregating

### Boosting Methods
- **Gradient Boosting:** Sequential error minimization
- **AdaBoost:** Adaptive weight adjustment
- **XGBoost:** Optimized gradient boosting
- **LightGBM:** Fast gradient boosting

### Meta-Ensemble Methods
- **Voting:** Hard/soft voting with diverse models
- **Stacking:** Meta-learner trained on base predictions
- **Blending:** Simplified stacking approach

## Advanced Features
- ✅ **Automatic Model Selection:** Intelligent base model choice
- ✅ **Ensemble Pruning:** Remove redundant models
- ✅ **Online Learning:** Adaptive ensembles for streaming data
- ✅ **Multi-objective Optimization:** Balance accuracy vs. complexity
- ✅ **Feature Importance Aggregation:** Combined feature analysis

## Performance Characteristics
- **Accuracy Improvements:** 2-15% over single models
- **Training Efficiency:** Parallel processing for most methods
- **Prediction Speed:** Optimized for production deployment
- **Scalability:** Handles datasets from thousands to millions of samples

## Integration & Compatibility
- ✅ **Scikit-learn Compatible:** Full estimator API compliance
- ✅ **External Library Support:** XGBoost, LightGBM, CatBoost integration
- ✅ **Parallel Processing:** Multi-core training and prediction
- ✅ **Model Persistence:** Efficient serialization and deployment
- ✅ **BaseModel Interface:** Consistent API across ML platform

## Quick Start
```python
from modeling.ensemble import EnsembleModel, EnsembleConfig, EnsembleMethod

config = EnsembleConfig(method=EnsembleMethod.RANDOM_FOREST)
ensemble = EnsembleModel(config)
ensemble.fit(X_train, y_train)
predictions = ensemble.predict(X_test)
```

## Use Cases
- **Competitive ML:** Kaggle competitions, benchmarks
- **Production ML:** Recommendation systems, fraud detection
- **Healthcare:** Disease diagnosis, treatment recommendation
- **Finance:** Credit scoring, risk assessment, trading
- **Manufacturing:** Quality control, predictive maintenance
- **NLP:** Text classification, sentiment analysis

## Best Practices
- **Maximize Diversity:** Use different algorithms and features
- **Balance Performance:** Individual model quality vs. ensemble diversity
- **Proper Validation:** Use appropriate cross-validation strategies
- **Computational Optimization:** Leverage parallelization and pruning

For detailed implementation examples and optimization guidelines, see the full documentation.
"""
    return summary.strip()


def export_info_json(filename: str = 'ensemble_methods_info.json') -> None:
    """Export complete module information to JSON file."""
    info_data = {
        'package_info': get_package_info(),
        'ensemble_method_comparison': get_ensemble_method_comparison(),
        'implementation_examples': get_implementation_examples(),
        'optimization_guidelines': get_optimization_guidelines(),
        'generated_at': datetime.now().isoformat()
    }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(info_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Ensemble methods module information exported to {filename}")
        print(f"✅ Module information exported to {filename}")
    except Exception as e:
        logger.error(f"Failed to export module information: {e}")
        print(f"❌ Export failed: {e}")


if __name__ == "__main__":
    # Generate and display summary when run directly
    print("🎯 Ensemble Methods Module Information")
    print("=" * 50)
    print(generate_info_summary())
    print("\n" + "=" * 50)
    
    # Optionally export to JSON
    export_choice = input("\nExport detailed information to JSON? (y/n): ").lower().strip()
    if export_choice in ['y', 'yes']:
        export_info_json()
    
    print("\n📚 Documentation complete!")
