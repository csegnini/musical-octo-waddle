"""
Classification Module Information Module.

This module provides comprehensive information about the classification module
capabilities, features, and usage guidelines for supervised classification methods
including logistic regression, SVM, decision trees, and ensemble methods.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


def get_package_info() -> Dict[str, Any]:
    """
    Get comprehensive classification module information.
    
    Returns:
        Dictionary containing complete module details
    """
    return {
        'module_name': 'Advanced Classification Framework',
        'version': '1.0.0',
        'description': 'Comprehensive classification framework with classical and modern supervised learning algorithms for binary and multiclass classification tasks',
        'author': 'AI ML Platform Team',
        'license': 'MIT',
        'created_date': '2025-07-27',
        'last_updated': datetime.now().isoformat(),
        'core_components': {
            'classification_models': {
                'file': '__init__.py',
                'lines_of_code': 875,
                'description': 'Advanced classification models with comprehensive evaluation and interpretation capabilities',
                'key_classes': ['LogisticRegressionModel', 'SVMModel', 'DecisionTreeModel', 'RandomForestModel', 'ClassificationModelFactory'],
                'features': [
                    '4+ core classification algorithms (Logistic Regression, SVM, Decision Trees, Random Forest)',
                    'Binary and multiclass classification support',
                    'Automatic model evaluation with comprehensive metrics',
                    'Feature importance analysis and interpretation',
                    'Probability estimation and confidence scoring',
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
                'logistic_regression': {
                    'description': 'Linear model for classification using logistic function',
                    'class_name': 'LogisticRegressionModel',
                    'algorithm_type': 'Linear classifier',
                    'strengths': ['Fast training', 'Interpretable coefficients', 'Probabilistic output', 'No hyperparameter tuning needed'],
                    'weaknesses': ['Assumes linear decision boundary', 'Sensitive to outliers', 'Requires feature scaling'],
                    'best_use_cases': ['Linear separable data', 'Need interpretability', 'Baseline models', 'Large datasets'],
                    'hyperparameters': {
                        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                        'C': 'float (inverse regularization strength)',
                        'solver': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'],
                        'multi_class': ['auto', 'ovr', 'multinomial'],
                        'class_weight': ['balanced', 'None', 'custom dict']
                    },
                    'complexity': 'O(n × p) training, O(p) prediction',
                    'output_types': ['class_labels', 'probabilities', 'log_odds']
                }
            },
            'kernel_methods': {
                'support_vector_machine': {
                    'description': 'Kernel-based classifier finding optimal decision boundary',
                    'class_name': 'SVMModel',
                    'algorithm_type': 'Kernel method',
                    'strengths': ['Effective in high dimensions', 'Memory efficient', 'Versatile kernels', 'Works with small datasets'],
                    'weaknesses': ['Slow on large datasets', 'No probabilistic output by default', 'Sensitive to feature scaling'],
                    'best_use_cases': ['High-dimensional data', 'Non-linear patterns', 'Text classification', 'Image recognition'],
                    'hyperparameters': {
                        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                        'C': 'float (regularization parameter)',
                        'gamma': ['scale', 'auto', 'float'],
                        'degree': 'int (polynomial kernel degree)',
                        'probability': 'bool (enable probability estimates)'
                    },
                    'complexity': 'O(n² to n³) training, O(n_sv × p) prediction',
                    'output_types': ['class_labels', 'probabilities (if enabled)', 'decision_function']
                }
            },
            'tree_based_methods': {
                'decision_tree': {
                    'description': 'Tree-structured classifier with interpretable rules',
                    'class_name': 'DecisionTreeModel',
                    'algorithm_type': 'Tree-based method',
                    'strengths': ['Highly interpretable', 'Handles mixed data types', 'No feature scaling needed', 'Fast prediction'],
                    'weaknesses': ['Prone to overfitting', 'Unstable (high variance)', 'Biased toward features with more levels'],
                    'best_use_cases': ['Need interpretability', 'Mixed data types', 'Rule extraction', 'Feature selection'],
                    'hyperparameters': {
                        'criterion': ['gini', 'entropy', 'log_loss'],
                        'max_depth': 'int or None',
                        'min_samples_split': 'int or float',
                        'min_samples_leaf': 'int or float',
                        'max_features': ['auto', 'sqrt', 'log2', 'int', 'float']
                    },
                    'complexity': 'O(n × p × log(n)) training, O(log(n)) prediction',
                    'output_types': ['class_labels', 'probabilities', 'feature_importance', 'tree_structure']
                },
                'random_forest': {
                    'description': 'Ensemble of decision trees with bootstrap aggregating',
                    'class_name': 'RandomForestModel',
                    'algorithm_type': 'Ensemble method',
                    'strengths': ['Reduces overfitting', 'Handles large datasets', 'Feature importance', 'Robust to outliers'],
                    'weaknesses': ['Less interpretable than single tree', 'Can overfit with noisy data', 'Memory intensive'],
                    'best_use_cases': ['General-purpose classification', 'Feature selection', 'Large datasets', 'Robust baseline'],
                    'hyperparameters': {
                        'n_estimators': 'int (number of trees)',
                        'max_depth': 'int or None',
                        'min_samples_split': 'int or float',
                        'min_samples_leaf': 'int or float',
                        'max_features': ['auto', 'sqrt', 'log2'],
                        'bootstrap': 'bool',
                        'oob_score': 'bool'
                    },
                    'complexity': 'O(n × p × log(n) × n_trees) training, O(log(n) × n_trees) prediction',
                    'output_types': ['class_labels', 'probabilities', 'feature_importance', 'oob_score']
                }
            }
        },
        'problem_types_supported': {
            'binary_classification': {
                'description': 'Two-class classification problems',
                'examples': ['Spam detection', 'Medical diagnosis', 'Fraud detection', 'Customer churn'],
                'metrics': ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC-ROC', 'AUC-PR'],
                'special_considerations': ['Class imbalance handling', 'Threshold optimization', 'Cost-sensitive learning'],
                'output_interpretation': 'Probability of positive class, binary decision'
            },
            'multiclass_classification': {
                'description': 'Multi-class classification problems with mutually exclusive classes',
                'examples': ['Image classification', 'Text categorization', 'Species identification', 'Product categorization'],
                'metrics': ['Accuracy', 'Macro/Micro/Weighted Precision', 'Macro/Micro/Weighted Recall', 'Macro/Micro/Weighted F1'],
                'special_considerations': ['One-vs-rest vs one-vs-one strategies', 'Class imbalance', 'Hierarchical classification'],
                'output_interpretation': 'Probability distribution over all classes'
            },
            'multilabel_classification': {
                'description': 'Multi-label classification where instances can belong to multiple classes',
                'examples': ['Tag prediction', 'Gene function prediction', 'Movie genre classification'],
                'metrics': ['Hamming loss', 'Jaccard score', 'F1-micro/macro', 'Label ranking'],
                'special_considerations': ['Label correlation', 'Label imbalance', 'Threshold selection'],
                'output_interpretation': 'Independent probabilities for each label'
            }
        },
        'evaluation_framework': {
            'performance_metrics': {
                'accuracy_metrics': {
                    'accuracy': {
                        'description': 'Fraction of correctly classified instances',
                        'formula': '(TP + TN) / (TP + TN + FP + FN)',
                        'range': '[0, 1] where 1 is perfect',
                        'best_for': 'Balanced datasets, overall performance assessment'
                    },
                    'balanced_accuracy': {
                        'description': 'Average of recall obtained on each class',
                        'formula': '(Sensitivity + Specificity) / 2',
                        'range': '[0, 1] where 1 is perfect',
                        'best_for': 'Imbalanced datasets'
                    }
                },
                'precision_recall_metrics': {
                    'precision': {
                        'description': 'Fraction of positive predictions that are correct',
                        'formula': 'TP / (TP + FP)',
                        'range': '[0, 1] where 1 is perfect',
                        'best_for': 'When false positives are costly'
                    },
                    'recall': {
                        'description': 'Fraction of positive instances that are correctly identified',
                        'formula': 'TP / (TP + FN)',
                        'range': '[0, 1] where 1 is perfect',
                        'best_for': 'When false negatives are costly'
                    },
                    'f1_score': {
                        'description': 'Harmonic mean of precision and recall',
                        'formula': '2 × (Precision × Recall) / (Precision + Recall)',
                        'range': '[0, 1] where 1 is perfect',
                        'best_for': 'Balance between precision and recall'
                    }
                },
                'probabilistic_metrics': {
                    'auc_roc': {
                        'description': 'Area under ROC curve',
                        'formula': 'Integral of TPR vs FPR curve',
                        'range': '[0, 1] where 1 is perfect',
                        'best_for': 'Ranking quality, threshold-independent'
                    },
                    'auc_pr': {
                        'description': 'Area under Precision-Recall curve',
                        'formula': 'Integral of Precision vs Recall curve',
                        'range': '[0, 1] where 1 is perfect',
                        'best_for': 'Imbalanced datasets'
                    },
                    'log_loss': {
                        'description': 'Logarithmic loss for probability predictions',
                        'formula': '-Σ(y×log(p) + (1-y)×log(1-p))',
                        'range': '[0, ∞] where 0 is perfect',
                        'best_for': 'Probability calibration assessment'
                    }
                }
            },
            'model_interpretation': {
                'feature_importance': {
                    'tree_based_importance': 'Gini importance from tree-based models',
                    'permutation_importance': 'Decrease in performance when feature is shuffled',
                    'coefficient_importance': 'Linear model coefficients (logistic regression)',
                    'shap_values': 'SHapley Additive exPlanations for individual predictions'
                },
                'decision_boundaries': {
                    'linear_boundaries': 'Hyperplane visualization for linear models',
                    'non_linear_boundaries': 'Complex decision boundaries for kernel/tree methods',
                    'confidence_regions': 'Areas of high/low prediction confidence'
                },
                'model_complexity': {
                    'vc_dimension': 'Vapnik-Chervonenkis dimension for model capacity',
                    'tree_depth': 'Maximum depth of decision trees',
                    'number_of_parameters': 'Total trainable parameters',
                    'regularization_strength': 'Impact of regularization on model complexity'
                }
            }
        },
        'advanced_features': {
            'class_imbalance_handling': {
                'description': 'Techniques for dealing with unequal class distributions',
                'sampling_methods': {
                    'oversampling': ['SMOTE', 'ADASYN', 'Random oversampling'],
                    'undersampling': ['Random undersampling', 'Tomek links', 'Edited nearest neighbors'],
                    'combined_methods': ['SMOTEENN', 'SMOTETomek']
                },
                'algorithmic_approaches': {
                    'class_weighting': 'Inverse frequency weighting for loss function',
                    'cost_sensitive_learning': 'Different misclassification costs',
                    'threshold_tuning': 'Optimal threshold selection for predictions',
                    'ensemble_methods': 'Balanced bagging and boosting'
                },
                'evaluation_considerations': {
                    'stratified_sampling': 'Maintain class distribution in train/test splits',
                    'appropriate_metrics': 'F1, AUC-PR, Balanced accuracy over simple accuracy',
                    'cross_validation': 'Stratified k-fold cross-validation'
                }
            },
            'hyperparameter_optimization': {
                'description': 'Systematic approach to finding optimal model parameters',
                'search_strategies': {
                    'grid_search': 'Exhaustive search over parameter grid',
                    'random_search': 'Random sampling from parameter distributions',
                    'bayesian_optimization': 'Sequential model-based optimization',
                    'evolutionary_algorithms': 'Genetic algorithm-based optimization'
                },
                'cross_validation': {
                    'k_fold': 'Standard k-fold cross-validation',
                    'stratified_k_fold': 'Maintains class distribution in folds',
                    'time_series_split': 'Time-aware splitting for temporal data',
                    'leave_one_out': 'Maximum data usage for small datasets'
                },
                'early_stopping': {
                    'validation_monitoring': 'Stop training when validation performance plateaus',
                    'patience_parameter': 'Number of epochs to wait for improvement',
                    'restoration': 'Restore best weights after early stopping'
                }
            },
            'ensemble_learning': {
                'description': 'Combining multiple models for improved performance',
                'bagging_methods': {
                    'bootstrap_aggregating': 'Training on bootstrap samples',
                    'random_subspaces': 'Random feature selection for diversity',
                    'extra_trees': 'Extremely randomized trees'
                },
                'boosting_methods': {
                    'adaboost': 'Adaptive boosting with instance reweighting',
                    'gradient_boosting': 'Sequential error correction',
                    'xgboost': 'Extreme gradient boosting with regularization'
                },
                'stacking_methods': {
                    'meta_learning': 'Learning to combine base model predictions',
                    'blending': 'Weighted combination of model outputs',
                    'multi_level_stacking': 'Hierarchical ensemble structures'
                }
            },
            'model_selection': {
                'description': 'Systematic approach to choosing the best model',
                'validation_strategies': {
                    'holdout_validation': 'Single train/validation/test split',
                    'cross_validation': 'Multiple train/validation cycles',
                    'nested_cross_validation': 'Unbiased performance estimation',
                    'bootstrap_validation': 'Resampling-based validation'
                },
                'selection_criteria': {
                    'performance_metrics': 'Primary metric optimization',
                    'model_complexity': 'Bias-variance tradeoff consideration',
                    'computational_efficiency': 'Training and prediction time constraints',
                    'interpretability_requirements': 'Business need for explainability'
                },
                'statistical_testing': {
                    'mcnemar_test': 'Comparing two classifiers on same dataset',
                    'friedman_test': 'Comparing multiple classifiers across datasets',
                    'wilcoxon_signed_rank': 'Non-parametric performance comparison'
                }
            }
        },
        'technical_specifications': {
            'performance_characteristics': {
                'scalability': {
                    'logistic_regression': 'Linear scaling with features and samples',
                    'svm': 'Quadratic to cubic scaling, memory efficient',
                    'decision_tree': 'Good scaling, log-linear prediction time',
                    'random_forest': 'Parallel training, slightly super-linear scaling'
                },
                'memory_requirements': {
                    'logistic_regression': 'O(p) for coefficients',
                    'svm': 'O(n_support_vectors × p) for support vectors',
                    'decision_tree': 'O(nodes) for tree structure',
                    'random_forest': 'O(n_trees × nodes_per_tree)'
                },
                'training_time': {
                    'small_datasets': '< 1,000 samples: < 1 second for all methods',
                    'medium_datasets': '1,000-100,000 samples: 1-60 seconds depending on method',
                    'large_datasets': '> 100,000 samples: Minutes to hours for complex methods'
                },
                'prediction_time': {
                    'real_time_requirements': 'Microseconds for simple models, milliseconds for complex',
                    'batch_processing': 'Highly vectorized operations for large batches',
                    'streaming_data': 'Online learning capabilities for adaptive models'
                }
            },
            'numerical_considerations': {
                'precision': 'Double-precision floating-point arithmetic',
                'stability': 'Numerically stable algorithms with convergence guarantees',
                'feature_scaling': 'Automatic detection of scaling requirements',
                'missing_value_handling': 'Multiple imputation and missing indicator methods'
            },
            'software_dependencies': {
                'core_requirements': ['scikit-learn', 'numpy', 'pandas', 'scipy'],
                'optional_dependencies': ['xgboost', 'lightgbm', 'catboost', 'shap'],
                'visualization': ['matplotlib', 'seaborn', 'plotly'],
                'hyperparameter_optimization': ['optuna', 'hyperopt', 'scikit-optimize']
            }
        },
        'integration_capabilities': {
            'data_preprocessing': {
                'feature_engineering': 'Automatic feature creation and selection',
                'data_cleaning': 'Outlier detection and missing value imputation',
                'feature_scaling': 'Standardization, normalization, robust scaling',
                'categorical_encoding': 'One-hot, label, target, and ordinal encoding'
            },
            'model_deployment': {
                'serialization': 'Joblib, pickle, and ONNX export formats',
                'rest_api': 'Flask/FastAPI integration for web services',
                'batch_processing': 'Distributed processing with Dask/Spark',
                'real_time_inference': 'Low-latency prediction services'
            },
            'monitoring_and_maintenance': {
                'performance_tracking': 'Continuous model performance monitoring',
                'data_drift_detection': 'Statistical tests for distribution changes',
                'model_retraining': 'Automated retraining pipelines',
                'a_b_testing': 'Framework for model comparison in production'
            }
        },
        'validation_framework': {
            'data_validation': {
                'description': 'Comprehensive input data validation and quality checks',
                'schema_validation': 'Type checking and format validation',
                'range_validation': 'Feature value range and distribution checks',
                'consistency_validation': 'Cross-feature consistency and logical constraints',
                'completeness_validation': 'Missing value patterns and data completeness'
            },
            'model_validation': {
                'description': 'Statistical validation of model performance and reliability',
                'performance_validation': 'Cross-validation and bootstrap confidence intervals',
                'stability_validation': 'Robustness to data perturbations',
                'fairness_validation': 'Bias detection across demographic groups',
                'calibration_validation': 'Probability calibration assessment'
            },
            'deployment_validation': {
                'description': 'Validation of model behavior in production environment',
                'integration_testing': 'End-to-end pipeline validation',
                'load_testing': 'Performance under high request volumes',
                'monitoring_validation': 'Alert systems and performance tracking',
                'rollback_procedures': 'Safe model deployment and rollback strategies'
            }
        }
    }


def get_algorithm_comparison() -> Dict[str, Dict[str, Any]]:
    """Get detailed comparison of classification algorithms."""
    return {
        'linear_vs_nonlinear': {
            'linear_methods': {
                'algorithms': ['Logistic Regression', 'Linear SVM'],
                'assumptions': ['Linear decision boundary', 'Feature independence', 'No multicollinearity'],
                'advantages': ['Fast training and prediction', 'Interpretable', 'Good baseline', 'Scales well'],
                'disadvantages': ['Limited to linear patterns', 'May underfit complex data'],
                'best_for': 'High-dimensional data, text classification, when interpretability is important',
                'typical_performance': 'Good for linearly separable data, baseline for complex problems'
            },
            'nonlinear_methods': {
                'algorithms': ['RBF SVM', 'Decision Trees', 'Random Forest'],
                'assumptions': ['Complex decision boundaries', 'Non-linear feature interactions'],
                'advantages': ['Captures complex patterns', 'No linearity assumption', 'Can model interactions'],
                'disadvantages': ['More prone to overfitting', 'Less interpretable', 'Longer training'],
                'best_for': 'Complex patterns, image/audio data, when accuracy is priority',
                'typical_performance': 'Better for complex data, risk of overfitting on small datasets'
            }
        },
        'parametric_vs_nonparametric': {
            'parametric_methods': {
                'algorithms': ['Logistic Regression'],
                'characteristics': ['Fixed number of parameters', 'Strong distributional assumptions'],
                'advantages': ['Fast convergence', 'Interpretable parameters', 'Good with small data'],
                'disadvantages': ['May be too restrictive', 'Assumption violations affect performance'],
                'data_requirements': 'Moderate sample sizes, well-behaved features'
            },
            'nonparametric_methods': {
                'algorithms': ['Decision Trees', 'Random Forest', 'SVM with RBF kernel'],
                'characteristics': ['Flexible parameter count', 'Minimal distributional assumptions'],
                'advantages': ['Very flexible', 'Adapts to data complexity', 'Robust to outliers'],
                'disadvantages': ['Needs more data', 'Risk of overfitting', 'Less interpretable'],
                'data_requirements': 'Large sample sizes for complex patterns'
            }
        },
        'single_vs_ensemble': {
            'single_models': {
                'algorithms': ['Logistic Regression', 'SVM', 'Single Decision Tree'],
                'advantages': ['Fast training', 'Interpretable', 'Simple deployment', 'Low memory'],
                'disadvantages': ['May have high bias or variance', 'Sensitive to data quality'],
                'use_cases': 'Simple problems, interpretability requirements, resource constraints'
            },
            'ensemble_models': {
                'algorithms': ['Random Forest', 'Gradient Boosting', 'Voting Classifiers'],
                'advantages': ['Better generalization', 'Robust to overfitting', 'Higher accuracy'],
                'disadvantages': ['More complex', 'Longer training', 'Less interpretable', 'More memory'],
                'use_cases': 'Complex problems, when accuracy is priority, sufficient computational resources'
            }
        },
        'probabilistic_vs_discriminative': {
            'probabilistic_models': {
                'algorithms': ['Logistic Regression (with calibration)', 'Naive Bayes'],
                'output': 'Well-calibrated probabilities',
                'advantages': ['Uncertainty quantification', 'Good for decision making', 'Handle class imbalance'],
                'disadvantages': ['May sacrifice some accuracy', 'Require probability calibration'],
                'best_for': 'Risk assessment, medical diagnosis, when probability estimates are important'
            },
            'discriminative_models': {
                'algorithms': ['SVM', 'Decision Trees'],
                'output': 'Class decisions (probabilities may not be well-calibrated)',
                'advantages': ['Focus on decision boundary', 'Often higher accuracy', 'Robust classification'],
                'disadvantages': ['Poor probability estimates', 'Less suitable for uncertainty quantification'],
                'best_for': 'Classification accuracy priority, when only decisions (not probabilities) needed'
            }
        }
    }


def get_implementation_examples() -> Dict[str, str]:
    """Get comprehensive implementation examples for classification models."""
    return {
        'logistic_regression_example': '''
# Logistic Regression Implementation
from modeling.classification import LogisticRegressionModel, create_logistic_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Method 1: Direct instantiation
model = LogisticRegressionModel(
    name="Customer_Churn_Classifier",
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    multi_class='auto',
    class_weight='balanced',  # Handle class imbalance
    random_state=42
)

# Method 2: Factory function
model = create_logistic_regression(
    name="Customer_Churn_Classifier",
    penalty='l1',  # L1 for feature selection
    C=0.1,
    solver='liblinear',  # Required for L1 penalty
    class_weight='balanced'
)

# Data preparation
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, stratify=target, random_state=42
)

# Feature scaling (important for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)

# Access model metrics
print(f"Training Accuracy: {model.get_metric('accuracy'):.4f}")
print(f"Training F1-Score: {model.get_metric('f1_score'):.4f}")
print(f"Training Precision: {model.get_metric('precision'):.4f}")
print(f"Training Recall: {model.get_metric('recall'):.4f}")

# Get model coefficients for interpretation
if hasattr(model._model, 'coef_'):
    feature_importance = dict(zip(feature_names, model._model.coef_[0]))
    sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
    print("\\nTop 5 Most Important Features:")
    for feature, coef in sorted_features[:5]:
        print(f"{feature}: {coef:.4f}")
''',

        'svm_example': '''
# Support Vector Machine Implementation
from modeling.classification import SVMModel, create_svm
from sklearn.preprocessing import StandardScaler

# Create SVM with RBF kernel for non-linear patterns
model = create_svm(
    name="Image_Classifier",
    kernel='rbf',
    C=10.0,  # Regularization parameter
    gamma='scale',  # Kernel coefficient
    probability=True,  # Enable probability estimates
    class_weight='balanced',
    random_state=42
)

# Alternative: Linear SVM for high-dimensional data
linear_svm = SVMModel(
    name="Text_Classifier",
    kernel='linear',
    C=1.0,
    probability=True,
    random_state=42
)

# Data preparation (scaling is crucial for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model.fit(X_train_scaled, y_train)

# Predictions and probabilities
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)  # Requires probability=True

# Access SVM-specific information
print(f"Number of Support Vectors: {model.get_metric('n_support_vectors')}")
support_vectors = model.get_support_vectors()
print(f"Support Vector Shape: {support_vectors.shape if support_vectors is not None else 'None'}")

# Model evaluation
from sklearn.metrics import classification_report, confusion_matrix
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))

# For multi-class problems, check decision function
if hasattr(model._model, 'decision_function'):
    decision_scores = model._model.decision_function(X_test_scaled)
    print(f"Decision function shape: {decision_scores.shape}")
''',

        'decision_tree_example': '''
# Decision Tree Implementation
from modeling.classification import DecisionTreeModel, create_decision_tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Create interpretable decision tree
model = create_decision_tree(
    name="Medical_Diagnosis_Tree",
    criterion='gini',  # or 'entropy' for information gain
    max_depth=5,  # Prevent overfitting
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',  # Random feature selection
    class_weight='balanced',
    random_state=42
)

# Train the model (no scaling needed for trees)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Get feature importance
feature_importance = model.get_feature_importance()
if feature_importance:
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    print("Top 5 Most Important Features:")
    for feature, importance in sorted_features[:5]:
        print(f"{feature}: {importance:.4f}")

# Tree-specific metrics
print(f"Tree Depth: {model.get_metric('tree_depth')}")
print(f"Number of Nodes: {model.get_metric('n_nodes')}")
print(f"Number of Leaves: {model.get_metric('n_leaves')}")

# Visualize the tree (for small trees)
plt.figure(figsize=(20, 10))
plot_tree(model._model, 
          feature_names=feature_names,
          class_names=model.classes_.astype(str),
          filled=True,
          max_depth=3)  # Limit visualization depth
plt.title("Decision Tree Visualization")
plt.show()

# Extract decision rules (for interpretation)
from sklearn.tree import export_text
tree_rules = export_text(model._model, feature_names=feature_names)
print("\\nDecision Tree Rules:")
print(tree_rules[:500])  # Print first 500 characters
''',

        'random_forest_example': '''
# Random Forest Implementation
from modeling.classification import RandomForestModel, create_random_forest
import numpy as np

# Create robust random forest
model = create_random_forest(
    name="Comprehensive_Classifier",
    n_estimators=100,  # Number of trees
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',  # sqrt(n_features) for classification
    bootstrap=True,
    oob_score=True,  # Out-of-bag score estimation
    n_jobs=-1,  # Use all processors
    class_weight='balanced',
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predictions with confidence
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Get prediction confidence (standard deviation across trees)
# Note: This requires accessing individual tree predictions
if hasattr(model._model, 'estimators_'):
    tree_predictions = np.array([tree.predict_proba(X_test) for tree in model._model.estimators_])
    prediction_std = np.std(tree_predictions, axis=0)
    confidence = 1 - np.max(prediction_std, axis=1)  # Higher std = lower confidence
    
    # Show predictions with confidence
    for i in range(min(10, len(y_test))):
        pred_class = y_pred[i]
        pred_proba = np.max(y_proba[i])
        pred_confidence = confidence[i]
        print(f"Sample {i}: Predicted={pred_class}, Probability={pred_proba:.3f}, Confidence={pred_confidence:.3f}")

# Feature importance analysis
feature_importance = model.get_feature_importance()
if feature_importance:
    importance_df = pd.DataFrame([
        {'feature': feature, 'importance': importance}
        for feature, importance in feature_importance.items()
    ]).sort_values('importance', ascending=False)
    
    print("\\nTop 10 Most Important Features:")
    print(importance_df.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df.head(10)['feature'], importance_df.head(10)['importance'])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# Access Random Forest specific metrics
if hasattr(model._model, 'oob_score_'):
    print(f"Out-of-Bag Score: {model._model.oob_score_:.4f}")

print(f"Number of Trees: {model._model.n_estimators}")
print(f"Training Accuracy: {model.get_metric('accuracy'):.4f}")
''',

        'comprehensive_evaluation': '''
# Comprehensive Model Evaluation and Comparison
from modeling.classification import (
    create_logistic_regression, create_svm, 
    create_decision_tree, create_random_forest
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import pandas as pd

# Create multiple models for comparison
models = {
    'Logistic Regression': create_logistic_regression(
        penalty='l2', C=1.0, class_weight='balanced'
    ),
    'SVM': create_svm(
        kernel='rbf', C=10.0, probability=True, class_weight='balanced'
    ),
    'Decision Tree': create_decision_tree(
        max_depth=10, min_samples_split=10, class_weight='balanced'
    ),
    'Random Forest': create_random_forest(
        n_estimators=100, max_depth=10, class_weight='balanced'
    )
}

# Cross-validation evaluation
cv_results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    # Perform cross-validation
    cv_scores = cross_val_score(model._model, X_train, y_train, cv=cv, scoring='f1_weighted')
    cv_results[name] = {
        'mean_cv_score': cv_scores.mean(),
        'std_cv_score': cv_scores.std(),
        'cv_scores': cv_scores
    }
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    # Test set evaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    cv_results[name].update({
        'test_accuracy': accuracy,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1,
        'y_pred': y_pred,
        'y_proba': y_proba
    })
    
    # ROC AUC for binary classification
    if len(np.unique(y_test)) == 2 and y_proba is not None:
        auc_score = roc_auc_score(y_test, y_proba[:, 1])
        cv_results[name]['test_auc'] = auc_score

# Create results DataFrame
results_df = pd.DataFrame([
    {
        'Model': name,
        'CV Score (Mean)': results['mean_cv_score'],
        'CV Score (Std)': results['std_cv_score'],
        'Test Accuracy': results['test_accuracy'],
        'Test F1': results['test_f1'],
        'Test AUC': results.get('test_auc', 'N/A')
    }
    for name, results in cv_results.items()
])

print("Model Comparison Results:")
print(results_df.round(4))

# Plot ROC curves for binary classification
if len(np.unique(y_test)) == 2:
    plt.figure(figsize=(10, 8))
    
    for name, results in cv_results.items():
        if results.get('y_proba') is not None:
            fpr, tpr, _ = roc_curve(y_test, results['y_proba'][:, 1])
            auc_score = results.get('test_auc', 0)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

# Feature importance comparison (for applicable models)
feature_importance_models = ['Decision Tree', 'Random Forest']
if any(model_name in feature_importance_models for model_name in models.keys()):
    plt.figure(figsize=(12, 8))
    
    for i, (name, model) in enumerate(models.items()):
        if name in feature_importance_models:
            importance = model.get_feature_importance()
            if importance:
                sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                features, importances = zip(*sorted_features[:10])
                
                plt.subplot(1, 2, i+1)
                plt.barh(features, importances)
                plt.xlabel('Feature Importance')
                plt.title(f'{name} - Top 10 Features')
                plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.show()
'''
    }


def get_best_practices() -> Dict[str, Any]:
    """Get comprehensive best practices for classification modeling."""
    return {
        'data_preparation': {
            'data_quality_checks': {
                'missing_values': 'Identify and handle missing values appropriately',
                'outlier_detection': 'Use statistical methods or domain knowledge to identify outliers',
                'data_consistency': 'Check for logical inconsistencies and data entry errors',
                'duplicate_detection': 'Remove or handle duplicate records',
                'feature_correlation': 'Identify and handle highly correlated features'
            },
            'feature_engineering': {
                'domain_knowledge': 'Incorporate domain expertise in feature creation',
                'interaction_features': 'Create meaningful feature interactions',
                'polynomial_features': 'Consider polynomial features for non-linear relationships',
                'binning_discretization': 'Convert continuous variables to categorical when appropriate',
                'time_based_features': 'Extract temporal patterns from datetime variables'
            },
            'preprocessing_pipeline': {
                'train_test_split': 'Stratified splitting to maintain class distribution',
                'feature_scaling': 'Standardize or normalize features for distance-based algorithms',
                'categorical_encoding': 'Choose appropriate encoding (one-hot, label, target)',
                'pipeline_creation': 'Use sklearn pipelines for reproducible preprocessing',
                'data_leakage_prevention': 'Fit preprocessing only on training data'
            }
        },
        'model_selection_guidelines': {
            'problem_assessment': {
                'data_size': 'Choose algorithms appropriate for dataset size',
                'feature_dimensionality': 'Consider curse of dimensionality',
                'class_distribution': 'Account for class imbalance',
                'interpretability_requirements': 'Balance accuracy vs. interpretability',
                'computational_constraints': 'Consider training and prediction time limits'
            },
            'algorithm_selection_flowchart': {
                'small_dataset_linear': 'Logistic Regression for interpretable baseline',
                'small_dataset_nonlinear': 'SVM with RBF kernel for complex patterns',
                'medium_dataset_general': 'Random Forest for robust performance',
                'large_dataset_linear': 'Logistic Regression with regularization',
                'large_dataset_complex': 'Ensemble methods or deep learning',
                'high_interpretability': 'Decision Trees or Logistic Regression',
                'maximum_accuracy': 'Ensemble methods (Random Forest, Gradient Boosting)'
            },
            'hyperparameter_tuning': {
                'search_strategy': 'Start with random search, refine with grid search',
                'cross_validation': 'Use stratified k-fold for reliable estimates',
                'early_stopping': 'Prevent overfitting in iterative algorithms',
                'validation_strategy': 'Separate validation set for final model selection',
                'computational_budget': 'Balance search thoroughness with available time'
            }
        },
        'training_best_practices': {
            'cross_validation_strategy': {
                'stratified_kfold': 'Maintain class distribution across folds',
                'time_series_cv': 'Use time-aware splitting for temporal data',
                'group_kfold': 'Prevent data leakage in grouped data',
                'repeated_cv': 'Multiple CV runs for more reliable estimates',
                'nested_cv': 'Unbiased performance estimation with hyperparameter tuning'
            },
            'class_imbalance_handling': {
                'sampling_techniques': 'SMOTE, undersampling, or combined methods',
                'class_weighting': 'Inverse frequency weighting in loss function',
                'threshold_tuning': 'Optimize decision threshold for business metrics',
                'cost_sensitive_learning': 'Assign different misclassification costs',
                'ensemble_approaches': 'Balanced bagging or boosting methods'
            },
            'overfitting_prevention': {
                'regularization': 'L1/L2 regularization for linear models',
                'early_stopping': 'Monitor validation performance',
                'feature_selection': 'Remove irrelevant or redundant features',
                'model_complexity': 'Choose appropriate model complexity',
                'ensemble_methods': 'Combine multiple models to reduce variance'
            }
        },
        'evaluation_best_practices': {
            'metric_selection': {
                'balanced_data': 'Accuracy, F1-score are generally appropriate',
                'imbalanced_data': 'Precision, Recall, F1, AUC-PR preferred over accuracy',
                'business_context': 'Choose metrics aligned with business objectives',
                'multi_class': 'Macro, micro, and weighted averages for comprehensive view',
                'probability_calibration': 'Brier score, calibration plots for probability quality'
            },
            'validation_procedures': {
                'holdout_testing': 'Reserve independent test set for final evaluation',
                'stratified_sampling': 'Maintain class distribution in all splits',
                'temporal_validation': 'Time-based splits for time-dependent data',
                'statistical_significance': 'McNemar\'s test for comparing classifiers',
                'confidence_intervals': 'Bootstrap confidence intervals for metrics'
            },
            'error_analysis': {
                'confusion_matrix': 'Detailed breakdown of classification errors',
                'error_patterns': 'Identify systematic misclassification patterns',
                'feature_analysis': 'Analyze feature importance and contributions',
                'instance_analysis': 'Examine misclassified instances for insights',
                'boundary_analysis': 'Visualize decision boundaries when possible'
            }
        },
        'deployment_considerations': {
            'model_monitoring': {
                'performance_tracking': 'Continuous monitoring of key metrics',
                'data_drift_detection': 'Statistical tests for input distribution changes',
                'prediction_monitoring': 'Track prediction distributions and patterns',
                'alert_systems': 'Automated alerts for performance degradation',
                'retraining_triggers': 'Criteria for when to retrain models'
            },
            'production_readiness': {
                'model_versioning': 'Track model versions and configurations',
                'a_b_testing': 'Gradual rollout and comparison frameworks',
                'rollback_procedures': 'Quick rollback to previous model versions',
                'scalability_testing': 'Load testing for high-volume predictions',
                'latency_optimization': 'Optimize prediction speed for real-time requirements'
            },
            'maintenance_procedures': {
                'periodic_evaluation': 'Regular assessment of model performance',
                'data_quality_monitoring': 'Ongoing data quality checks',
                'feature_drift_tracking': 'Monitor changes in feature distributions',
                'model_updates': 'Systematic approach to model updates and improvements',
                'documentation_maintenance': 'Keep documentation current with changes'
            }
        },
        'common_pitfalls_and_solutions': {
            'data_leakage': {
                'problem': 'Information from the future or target leaks into features',
                'prevention': 'Careful feature engineering, proper time-aware splits',
                'detection': 'Suspiciously high performance, feature importance analysis'
            },
            'overfitting': {
                'problem': 'Model memorizes training data, poor generalization',
                'prevention': 'Regularization, cross-validation, simpler models',
                'detection': 'Large gap between training and validation performance'
            },
            'underfitting': {
                'problem': 'Model too simple to capture underlying patterns',
                'solutions': 'More complex models, feature engineering, less regularization',
                'detection': 'Poor performance on both training and validation sets'
            },
            'class_imbalance_ignorance': {
                'problem': 'Ignoring unequal class distributions',
                'solutions': 'Appropriate metrics, sampling techniques, class weighting',
                'detection': 'High accuracy but poor precision/recall for minority class'
            },
            'improper_validation': {
                'problem': 'Biased performance estimates',
                'solutions': 'Proper cross-validation, independent test sets, stratification',
                'detection': 'Results too good to be true, inconsistent performance'
            }
        }
    }


def get_performance_benchmarks() -> Dict[str, Any]:
    """Get performance benchmarks and computational expectations."""
    return {
        'algorithm_performance_characteristics': {
            'logistic_regression': {
                'training_complexity': 'O(n × p × i) where i is iterations',
                'prediction_complexity': 'O(p)',
                'memory_usage': 'O(p) for model parameters',
                'typical_training_time': {
                    'small_dataset': '< 1 second (n < 1K)',
                    'medium_dataset': '1-10 seconds (n = 1K-100K)',
                    'large_dataset': '10-300 seconds (n > 100K)'
                },
                'scalability': 'Excellent - linear scaling',
                'best_performance_conditions': 'Linearly separable data, many features'
            },
            'svm': {
                'training_complexity': 'O(n² to n³) depending on kernel and solver',
                'prediction_complexity': 'O(n_sv × p) where n_sv is support vectors',
                'memory_usage': 'O(n_sv × p) for support vectors',
                'typical_training_time': {
                    'small_dataset': '1-5 seconds (n < 1K)',
                    'medium_dataset': '30 seconds - 10 minutes (n = 1K-10K)',
                    'large_dataset': 'Hours (n > 10K, may be impractical)'
                },
                'scalability': 'Poor for large datasets',
                'best_performance_conditions': 'High-dimensional data, clear margins'
            },
            'decision_tree': {
                'training_complexity': 'O(n × p × log(n))',
                'prediction_complexity': 'O(log(n)) average case',
                'memory_usage': 'O(nodes) for tree structure',
                'typical_training_time': {
                    'small_dataset': '< 1 second (n < 1K)',
                    'medium_dataset': '1-30 seconds (n = 1K-100K)',
                    'large_dataset': '30 seconds - 10 minutes (n > 100K)'
                },
                'scalability': 'Good - near-linear scaling',
                'best_performance_conditions': 'Categorical features, interpretability needed'
            },
            'random_forest': {
                'training_complexity': 'O(n × p × log(n) × n_trees)',
                'prediction_complexity': 'O(log(n) × n_trees)',
                'memory_usage': 'O(nodes × n_trees)',
                'typical_training_time': {
                    'small_dataset': '1-5 seconds (n < 1K)',
                    'medium_dataset': '10 seconds - 5 minutes (n = 1K-100K)',
                    'large_dataset': '5-60 minutes (n > 100K)'
                },
                'scalability': 'Good with parallelization',
                'best_performance_conditions': 'Mixed data types, robust performance needed'
            }
        },
        'dataset_size_guidelines': {
            'small_datasets': {
                'sample_size': '< 1,000 samples',
                'recommended_algorithms': ['Logistic Regression', 'SVM', 'Decision Tree'],
                'special_considerations': ['Avoid overfitting', 'Simple models preferred', 'Cross-validation essential'],
                'typical_accuracy_range': '60-85% depending on problem complexity'
            },
            'medium_datasets': {
                'sample_size': '1,000 - 100,000 samples',
                'recommended_algorithms': ['Random Forest', 'SVM', 'Logistic Regression'],
                'special_considerations': ['Good for most algorithms', 'Hyperparameter tuning beneficial'],
                'typical_accuracy_range': '70-95% depending on problem complexity'
            },
            'large_datasets': {
                'sample_size': '> 100,000 samples',
                'recommended_algorithms': ['Logistic Regression', 'Random Forest', 'Gradient Boosting'],
                'special_considerations': ['Computational efficiency important', 'Simple models often sufficient'],
                'typical_accuracy_range': '75-98% depending on problem complexity'
            }
        },
        'feature_dimensionality_impact': {
            'low_dimensional': {
                'feature_count': '< 10 features',
                'challenges': ['May need feature engineering', 'Risk of underfitting'],
                'opportunities': ['Easy visualization', 'Good interpretability'],
                'recommended_approaches': ['Feature interactions', 'Polynomial features']
            },
            'medium_dimensional': {
                'feature_count': '10 - 1,000 features',
                'challenges': ['Moderate curse of dimensionality'],
                'opportunities': ['Rich feature space', 'Good for most algorithms'],
                'recommended_approaches': ['Feature selection', 'Regularization']
            },
            'high_dimensional': {
                'feature_count': '> 1,000 features',
                'challenges': ['Curse of dimensionality', 'Overfitting risk'],
                'opportunities': ['Rich information content'],
                'recommended_approaches': ['Dimensionality reduction', 'L1 regularization', 'Feature selection']
            }
        },
        'class_distribution_impact': {
            'balanced_classes': {
                'distribution': '40-60% for binary, roughly equal for multiclass',
                'recommended_metrics': ['Accuracy', 'F1-score', 'AUC-ROC'],
                'special_handling': 'Standard algorithms work well',
                'expected_performance': 'Algorithm-dependent baseline performance'
            },
            'moderate_imbalance': {
                'distribution': '20-40% minority class for binary',
                'recommended_metrics': ['Precision', 'Recall', 'F1-score', 'AUC-PR'],
                'special_handling': ['Class weighting', 'Threshold tuning'],
                'expected_performance': '5-10% reduction in minority class recall'
            },
            'severe_imbalance': {
                'distribution': '< 20% minority class for binary',
                'recommended_metrics': ['Precision', 'Recall', 'F1-score', 'AUC-PR'],
                'special_handling': ['SMOTE', 'Undersampling', 'Cost-sensitive learning'],
                'expected_performance': 'Significant challenge, specialized techniques required'
            }
        }
    }


def generate_info_summary() -> str:
    """Generate a comprehensive summary of the classification module."""
    info = get_package_info()
    
    summary = f"""
# Classification Module Summary

## Overview
{info['description']}

**Version:** {info['version']}
**Last Updated:** {info['last_updated']}

## Core Algorithms
- **Logistic Regression:** Fast, interpretable linear classifier with regularization
- **Support Vector Machine:** Kernel-based classifier for high-dimensional and non-linear data
- **Decision Tree:** Interpretable tree-based classifier with rule extraction
- **Random Forest:** Robust ensemble method combining multiple decision trees

## Key Capabilities
- **Binary & Multiclass Classification** with automatic problem type detection
- **Comprehensive Evaluation Metrics** (Accuracy, Precision, Recall, F1, AUC)
- **Feature Importance Analysis** for model interpretation
- **Probability Estimation** with calibration support
- **Class Imbalance Handling** with weighting and sampling techniques
- **Hyperparameter Optimization** support with cross-validation

## Problem Types Supported
- **Binary Classification:** Two-class problems (spam detection, medical diagnosis)
- **Multiclass Classification:** Multiple mutually exclusive classes
- **Imbalanced Classification:** Handling unequal class distributions

## Advanced Features
- **Model Interpretation:** Feature importance, SHAP values, decision rules
- **Ensemble Learning:** Bootstrap aggregating, voting classifiers
- **Cross-Validation:** Stratified k-fold with performance estimation
- **Model Selection:** Automated algorithm comparison and selection

## Integration
- ✅ Scikit-learn Backend
- ✅ BaseModel Interface Compliance
- ✅ Pandas DataFrame Support
- ✅ Automatic Metric Calculation
- ✅ Model Persistence (Joblib/Pickle)
- ✅ Production Deployment Ready

## Performance Characteristics
- **Logistic Regression:** O(n×p) training, excellent scalability
- **SVM:** O(n²-n³) training, best for small-medium datasets
- **Decision Tree:** O(n×p×log(n)) training, fast prediction
- **Random Forest:** Parallel training, robust performance

## Quick Start
```python
from modeling.classification import create_random_forest

# Create and train model
model = create_random_forest(n_estimators=100, class_weight='balanced')
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Get metrics and feature importance
accuracy = model.get_metric('accuracy')
importance = model.get_feature_importance()
```

For detailed usage examples and advanced classification techniques, see the full documentation.
"""
    return summary.strip()


def export_info_json(filename: str = 'classification_info.json') -> None:
    """Export complete module information to JSON file."""
    info_data = {
        'package_info': get_package_info(),
        'algorithm_comparison': get_algorithm_comparison(),
        'implementation_examples': get_implementation_examples(),
        'best_practices': get_best_practices(),
        'performance_benchmarks': get_performance_benchmarks(),
        'generated_at': datetime.now().isoformat()
    }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(info_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Classification module information exported to {filename}")
        print(f"✅ Module information exported to {filename}")
    except Exception as e:
        logger.error(f"Failed to export module information: {e}")
        print(f"❌ Export failed: {e}")


if __name__ == "__main__":
    # Generate and display summary when run directly
    print("🎯 Classification Module Information")
    print("=" * 60)
    print(generate_info_summary())
    print("\n" + "=" * 60)
    
    # Optionally export to JSON
    export_choice = input("\nExport detailed information to JSON? (y/n): ").lower().strip()
    if export_choice in ['y', 'yes']:
        export_info_json()
    
    print("\n📚 Documentation complete!")
