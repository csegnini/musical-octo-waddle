"""
Deep Learning Module Information Module.

This module provides comprehensive information about the deep learning module
capabilities, features, and usage guidelines for advanced neural network architectures.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


def get_package_info() -> Dict[str, Any]:
    """
    Get comprehensive deep learning module information.
    
    Returns:
        Dictionary containing complete module details
    """
    return {
        'module_name': 'Advanced Deep Learning Framework',
        'version': '1.0.0',
        'description': 'Comprehensive deep learning framework with TensorFlow/Keras integration for classification, regression, and advanced neural architectures',
        'author': 'AI ML Platform Team',
        'license': 'MIT',
        'created_date': '2025-07-27',
        'last_updated': datetime.now().isoformat(),
        'core_components': {
            'deep_learning_models': {
                'file': '__init__.py',
                'lines_of_code': 800,
                'description': 'Advanced deep learning models with multiple architectures and automatic optimization',
                'key_classes': ['DeepLearningModel', 'DeepLearningConfig', 'ArchitectureType'],
                'features': [
                    '8+ neural network architectures (Dense, CNN, RNN, LSTM, GRU, Bidirectional, Autoencoder, Custom)',
                    'Automatic hyperparameter optimization with grid/random search',
                    'Advanced regularization (Dropout, L1/L2, Batch Normalization)',
                    'Flexible activation functions and optimizers',
                    'Early stopping and learning rate scheduling',
                    'Model checkpointing and state management',
                    'GPU acceleration support',
                    'Comprehensive evaluation and visualization'
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
        'supported_architectures': {
            'feedforward_networks': {
                'dense': {
                    'description': 'Fully connected deep neural networks',
                    'use_cases': ['Tabular data', 'Feature learning', 'General classification/regression'],
                    'layer_types': ['Dense', 'Dropout', 'BatchNormalization'],
                    'typical_structure': '3-5 hidden layers with 64-512 neurons each'
                }
            },
            'convolutional_networks': {
                'cnn': {
                    'description': 'Convolutional Neural Networks for spatial data',
                    'use_cases': ['Image classification', 'Computer vision', 'Spatial pattern recognition'],
                    'layer_types': ['Conv2D', 'MaxPooling2D', 'Flatten', 'Dense'],
                    'typical_structure': 'Conv layers → Pooling → Dense classifier'
                }
            },
            'recurrent_networks': {
                'rnn': {
                    'description': 'Basic Recurrent Neural Networks',
                    'use_cases': ['Sequence modeling', 'Time series prediction'],
                    'layer_types': ['SimpleRNN', 'Dense'],
                    'typical_structure': 'RNN layers → Dense output'
                },
                'lstm': {
                    'description': 'Long Short-Term Memory networks',
                    'use_cases': ['Long sequence modeling', 'Time series forecasting', 'NLP tasks'],
                    'layer_types': ['LSTM', 'Dense', 'Dropout'],
                    'typical_structure': 'LSTM layers → Dense output with optional return_sequences'
                },
                'gru': {
                    'description': 'Gated Recurrent Unit networks',
                    'use_cases': ['Efficient sequence modeling', 'Time series analysis'],
                    'layer_types': ['GRU', 'Dense', 'Dropout'],
                    'typical_structure': 'GRU layers → Dense output'
                },
                'bidirectional': {
                    'description': 'Bidirectional recurrent networks',
                    'use_cases': ['Sequence classification', 'Context-aware predictions'],
                    'layer_types': ['Bidirectional(LSTM/GRU)', 'Dense'],
                    'typical_structure': 'Bidirectional RNN → Dense classifier'
                }
            },
            'specialized_networks': {
                'autoencoder': {
                    'description': 'Autoencoder networks for unsupervised learning',
                    'use_cases': ['Dimensionality reduction', 'Feature learning', 'Anomaly detection'],
                    'layer_types': ['Encoder', 'Decoder', 'Dense'],
                    'typical_structure': 'Input → Encoder → Latent → Decoder → Reconstruction'
                },
                'custom': {
                    'description': 'User-defined custom architectures',
                    'use_cases': ['Specialized problems', 'Research applications', 'Hybrid models'],
                    'layer_types': 'User-configurable',
                    'typical_structure': 'Custom layer sequences and connections'
                }
            }
        },
        'optimization_capabilities': {
            'hyperparameter_optimization': {
                'description': 'Automatic hyperparameter tuning for optimal performance',
                'methods': {
                    'grid_search': {
                        'description': 'Exhaustive search over parameter grid',
                        'parameters': ['learning_rate', 'batch_size', 'hidden_units', 'dropout_rate'],
                        'suitable_for': 'Small parameter spaces, guaranteed optimal solution'
                    },
                    'random_search': {
                        'description': 'Random sampling of parameter combinations',
                        'parameters': ['learning_rate', 'batch_size', 'layers', 'neurons_per_layer'],
                        'suitable_for': 'Large parameter spaces, efficient exploration'
                    }
                },
                'optimization_targets': [
                    'Model accuracy/performance',
                    'Training time efficiency',
                    'Model complexity balance',
                    'Generalization capability'
                ]
            },
            'training_optimization': {
                'early_stopping': {
                    'description': 'Prevent overfitting with validation-based stopping',
                    'monitors': ['val_loss', 'val_accuracy', 'val_mse'],
                    'parameters': ['patience', 'min_delta', 'restore_best_weights']
                },
                'learning_rate_scheduling': {
                    'description': 'Dynamic learning rate adjustment during training',
                    'schedulers': ['ReduceLROnPlateau', 'ExponentialDecay', 'StepDecay'],
                    'benefits': ['Faster convergence', 'Better final performance', 'Training stability']
                },
                'regularization': {
                    'techniques': ['Dropout', 'L1 Regularization', 'L2 Regularization', 'Batch Normalization'],
                    'benefits': ['Overfitting prevention', 'Better generalization', 'Training stability']
                }
            }
        },
        'supported_problem_types': [
            {
                'name': 'Binary Classification',
                'description': 'Two-class prediction problems',
                'output_activation': 'sigmoid',
                'loss_function': 'binary_crossentropy',
                'metrics': ['accuracy', 'precision', 'recall', 'f1_score'],
                'typical_applications': 'Fraud detection, medical diagnosis, spam detection'
            },
            {
                'name': 'Multi-class Classification',
                'description': 'Multiple class prediction problems',
                'output_activation': 'softmax',
                'loss_function': 'categorical_crossentropy',
                'metrics': ['accuracy', 'top_k_accuracy', 'precision', 'recall'],
                'typical_applications': 'Image classification, text categorization, pattern recognition'
            },
            {
                'name': 'Multi-label Classification',
                'description': 'Multiple independent binary predictions',
                'output_activation': 'sigmoid',
                'loss_function': 'binary_crossentropy',
                'metrics': ['accuracy', 'hamming_loss', 'jaccard_score'],
                'typical_applications': 'Tag prediction, multi-symptom diagnosis, feature detection'
            },
            {
                'name': 'Regression',
                'description': 'Continuous value prediction problems',
                'output_activation': 'linear',
                'loss_function': 'mse',
                'metrics': ['mae', 'mse', 'rmse', 'r2_score'],
                'typical_applications': 'Price prediction, forecasting, value estimation'
            },
            {
                'name': 'Time Series Forecasting',
                'description': 'Sequential data prediction with temporal dependencies',
                'architectures': ['LSTM', 'GRU', 'Bidirectional'],
                'special_features': ['Sequence-to-sequence', 'Multi-step ahead prediction'],
                'typical_applications': 'Stock price forecasting, demand prediction, weather forecasting'
            }
        ],
        'advanced_features': {
            'model_architecture_search': {
                'description': 'Automated architecture optimization',
                'search_space': {
                    'layer_types': ['Dense', 'LSTM', 'GRU', 'Conv2D'],
                    'layer_sizes': 'Configurable ranges',
                    'depth': '1-10 layers',
                    'connections': 'Sequential and skip connections'
                },
                'optimization_strategy': 'Performance-based selection with cross-validation'
            },
            'transfer_learning': {
                'description': 'Leverage pre-trained models for faster training',
                'supported_models': 'TensorFlow/Keras pre-trained models',
                'fine_tuning': 'Layer-wise learning rate control',
                'applications': 'Domain adaptation, few-shot learning'
            },
            'ensemble_integration': {
                'description': 'Integration with ensemble methods',
                'ensemble_types': ['Voting', 'Bagging', 'Boosting', 'Stacking'],
                'combination_strategies': 'Model averaging, weighted voting',
                'performance_boost': '2-5% typical improvement'
            },
            'interpretability': {
                'feature_importance': 'Layer-wise activation analysis',
                'attention_mechanisms': 'Built-in attention layers for interpretable models',
                'visualization': 'Training progress and architecture visualization'
            }
        },
        'performance_metrics': {
            'training_metrics': [
                'Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy'
            ],
            'classification_metrics': [
                'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC',
                'Confusion Matrix', 'Classification Report'
            ],
            'regression_metrics': [
                'Mean Absolute Error (MAE)', 'Mean Squared Error (MSE)',
                'Root Mean Squared Error (RMSE)', 'R-squared Score', 'Mean Absolute Percentage Error (MAPE)'
            ],
            'training_efficiency': [
                'Training Time', 'Convergence Speed', 'Memory Usage', 'GPU Utilization'
            ],
            'model_complexity': [
                'Number of Parameters', 'Model Size', 'Inference Time', 'FLOPS'
            ]
        },
        'technical_specifications': {
            'performance': {
                'training_speed': 'GPU-accelerated training (10-100x speedup)',
                'inference_time': '<1ms for simple models, <10ms for complex models',
                'memory_efficiency': 'Batch processing for memory optimization',
                'scalability': 'Supports models with millions of parameters'
            },
            'hardware_requirements': {
                'cpu_minimum': '4+ cores recommended',
                'ram_minimum': '8GB RAM (16GB+ for large models)',
                'gpu_support': 'CUDA-compatible GPUs (optional but recommended)',
                'storage': 'SSD recommended for large datasets'
            },
            'software_compatibility': {
                'python_version': '3.7+',
                'tensorflow_version': '2.8+',
                'required_dependencies': ['tensorflow', 'pandas', 'numpy', 'scikit-learn'],
                'optional_dependencies': ['matplotlib', 'seaborn', 'plotly', 'tensorboard'],
                'os_support': ['Windows', 'Linux', 'macOS']
            },
            'data_compatibility': {
                'input_formats': ['numpy arrays', 'pandas DataFrames', 'TensorFlow datasets'],
                'data_types': ['Numerical', 'Categorical (encoded)', 'Sequential', 'Image'],
                'preprocessing': 'Automatic scaling and normalization options',
                'batch_processing': 'Configurable batch sizes for memory management'
            }
        },
        'integration_capabilities': {
            'ml_framework_integration': {
                'tensorflow_keras': 'Native TensorFlow/Keras backend',
                'scikit_learn': 'Compatible with sklearn pipelines and metrics',
                'pandas_integration': 'Direct DataFrame input/output support',
                'numpy_optimization': 'Vectorized operations for performance'
            },
            'deployment_options': {
                'model_export': ['SavedModel format', 'HDF5', 'TensorFlow Lite'],
                'serving_platforms': ['TensorFlow Serving', 'REST APIs', 'Edge deployment'],
                'cloud_deployment': 'Compatible with major cloud ML platforms',
                'containerization': 'Docker-ready model packaging'
            },
            'monitoring_logging': {
                'tensorboard_integration': 'Real-time training visualization',
                'custom_callbacks': 'Extensible monitoring framework',
                'model_versioning': 'Automatic model checkpoint management',
                'experiment_tracking': 'Comprehensive training history logging'
            }
        },
        'usage_patterns': {
            'rapid_prototyping': {
                'description': 'Quick model development and testing',
                'typical_workflow': 'Config → Fit → Evaluate → Iterate',
                'time_to_first_model': '5-10 minutes',
                'use_cases': 'Proof of concept, baseline establishment'
            },
            'production_development': {
                'description': 'Robust model development for deployment',
                'typical_workflow': 'Research → Optimize → Validate → Deploy',
                'optimization_focus': 'Performance, reliability, interpretability',
                'use_cases': 'Production systems, critical applications'
            },
            'research_experimentation': {
                'description': 'Advanced model research and development',
                'typical_workflow': 'Hypothesize → Experiment → Analyze → Publish',
                'flexibility_focus': 'Custom architectures, novel techniques',
                'use_cases': 'Academic research, algorithm development'
            }
        },
        'best_practices': {
            'data_preparation': {
                'normalization': 'StandardScaler or MinMaxScaler recommended',
                'validation_split': '20-30% of data for validation',
                'data_quality': 'Clean, representative datasets for best results',
                'feature_engineering': 'Domain-specific feature creation'
            },
            'model_selection': {
                'architecture_choice': 'Match architecture to problem type and data characteristics',
                'complexity_balance': 'Start simple, increase complexity as needed',
                'regularization': 'Use dropout and L1/L2 regularization to prevent overfitting',
                'hyperparameter_tuning': 'Systematic optimization for best performance'
            },
            'training_strategy': {
                'batch_size': '32-256 typical range, adjust based on memory',
                'learning_rate': 'Start with 0.001, adjust based on convergence',
                'early_stopping': 'Monitor validation metrics to prevent overfitting',
                'checkpointing': 'Save best models during training'
            },
            'evaluation_validation': {
                'cross_validation': 'Use k-fold CV for robust performance estimation',
                'holdout_testing': 'Separate test set for final model evaluation',
                'metric_selection': 'Choose metrics appropriate for problem type',
                'bias_detection': 'Test for fairness and bias in predictions'
            }
        },
        'error_handling': {
            'common_issues': {
                'overfitting': 'Use regularization, early stopping, more data',
                'underfitting': 'Increase model complexity, reduce regularization',
                'slow_convergence': 'Adjust learning rate, use learning rate scheduling',
                'memory_errors': 'Reduce batch size, use gradient accumulation'
            },
            'debugging_tools': {
                'training_curves': 'Monitor loss and accuracy evolution',
                'gradient_monitoring': 'Track gradient flow and magnitudes',
                'layer_outputs': 'Inspect intermediate layer activations',
                'performance_profiling': 'Identify computational bottlenecks'
            },
            'recovery_strategies': {
                'checkpoint_restoration': 'Resume training from saved states',
                'learning_rate_adjustment': 'Dynamic learning rate modification',
                'architecture_modification': 'Iterative model improvement',
                'data_augmentation': 'Expand training data for better generalization'
            }
        }
    }


def get_architecture_breakdown() -> Dict[str, Dict[str, Any]]:
    """Get detailed breakdown of all supported architectures."""
    return {
        'dense_networks': {
            'description': 'Fully connected feedforward neural networks',
            'advantages': ['Simple to implement', 'Fast training', 'Universal approximation'],
            'disadvantages': ['No spatial/temporal awareness', 'Parameter intensive'],
            'best_for': ['Tabular data', 'Feature learning', 'Non-spatial problems'],
            'typical_config': {
                'hidden_layers': '3-5 layers',
                'neurons_per_layer': '64-512 neurons',
                'activation': 'ReLU or LeakyReLU',
                'dropout_rate': '0.2-0.5'
            }
        },
        'convolutional_networks': {
            'description': 'Networks with convolutional layers for spatial data',
            'advantages': ['Spatial feature extraction', 'Parameter sharing', 'Translation invariance'],
            'disadvantages': ['Requires spatial data', 'More complex to tune'],
            'best_for': ['Image data', 'Spatial patterns', 'Computer vision'],
            'typical_config': {
                'conv_layers': '2-4 layers',
                'filters': '32-256 filters per layer',
                'kernel_size': '3x3 or 5x5',
                'pooling': 'MaxPooling2D'
            }
        },
        'recurrent_networks': {
            'description': 'Networks with memory for sequential data',
            'advantages': ['Temporal modeling', 'Variable length sequences', 'Memory retention'],
            'disadvantages': ['Vanishing gradients', 'Sequential processing'],
            'best_for': ['Time series', 'Sequences', 'Temporal dependencies'],
            'typical_config': {
                'rnn_layers': '1-3 layers',
                'units': '50-200 units per layer',
                'dropout': '0.2-0.3',
                'return_sequences': 'True for stacking'
            }
        },
        'lstm_networks': {
            'description': 'Long Short-Term Memory networks for long sequences',
            'advantages': ['Long-term memory', 'Gradient flow control', 'Robust to vanishing gradients'],
            'disadvantages': ['Computational complexity', 'More parameters'],
            'best_for': ['Long sequences', 'Time series forecasting', 'NLP tasks'],
            'typical_config': {
                'lstm_layers': '1-2 layers',
                'units': '50-128 units per layer',
                'dropout': '0.2-0.3',
                'recurrent_dropout': '0.2'
            }
        },
        'gru_networks': {
            'description': 'Gated Recurrent Unit networks - simpler alternative to LSTM',
            'advantages': ['Faster than LSTM', 'Fewer parameters', 'Good performance'],
            'disadvantages': ['Less expressive than LSTM', 'Still sequential'],
            'best_for': ['Efficient sequence modeling', 'Resource-constrained environments'],
            'typical_config': {
                'gru_layers': '1-2 layers',
                'units': '50-128 units per layer',
                'dropout': '0.2-0.3',
                'reset_after': 'True'
            }
        },
        'bidirectional_networks': {
            'description': 'Process sequences in both forward and backward directions',
            'advantages': ['Full context utilization', 'Better sequence understanding'],
            'disadvantages': ['Double computation', 'Cannot be used for real-time prediction'],
            'best_for': ['Sequence classification', 'Context-dependent tasks'],
            'typical_config': {
                'base_rnn': 'LSTM or GRU',
                'merge_mode': 'concat',
                'units': '25-64 per direction',
                'return_sequences': 'False for classification'
            }
        },
        'autoencoder_networks': {
            'description': 'Networks for unsupervised representation learning',
            'advantages': ['Dimensionality reduction', 'Feature learning', 'Anomaly detection'],
            'disadvantages': ['Unsupervised only', 'Reconstruction-focused'],
            'best_for': ['Feature extraction', 'Anomaly detection', 'Data compression'],
            'typical_config': {
                'encoder_layers': '2-3 layers',
                'latent_dimension': '10-100 dimensions',
                'decoder_layers': 'Mirror of encoder',
                'activation': 'ReLU + sigmoid output'
            }
        }
    }


def get_optimization_strategies() -> Dict[str, Any]:
    """Get detailed optimization and tuning strategies."""
    return {
        'hyperparameter_optimization': {
            'grid_search': {
                'description': 'Exhaustive search over parameter combinations',
                'parameters': {
                    'learning_rate': [0.001, 0.01, 0.1],
                    'batch_size': [16, 32, 64, 128],
                    'hidden_units': [64, 128, 256, 512],
                    'dropout_rate': [0.1, 0.2, 0.3, 0.5],
                    'l2_regularization': [0.001, 0.01, 0.1]
                },
                'pros': ['Guaranteed to find best combination in search space', 'Reproducible results'],
                'cons': ['Computationally expensive', 'Limited to small search spaces'],
                'best_for': 'Small parameter spaces, critical applications'
            },
            'random_search': {
                'description': 'Random sampling from parameter distributions',
                'parameters': {
                    'learning_rate': 'log-uniform(1e-4, 1e-1)',
                    'batch_size': 'choice([16, 32, 64, 128, 256])',
                    'hidden_layers': 'randint(2, 6)',
                    'neurons_per_layer': 'randint(32, 512)',
                    'dropout_rate': 'uniform(0.1, 0.5)'
                },
                'pros': ['More efficient than grid search', 'Can explore larger spaces'],
                'cons': ['No guarantee of finding optimal', 'May miss important regions'],
                'best_for': 'Large parameter spaces, exploratory analysis'
            },
            'optimization_tips': {
                'learning_rate': 'Start with 0.001, use learning rate finder',
                'batch_size': 'Powers of 2, balance memory and gradient noise',
                'architecture': 'Start simple, gradually increase complexity',
                'regularization': 'Add when overfitting is observed'
            }
        },
        'training_optimization': {
            'early_stopping': {
                'monitor': 'val_loss',
                'patience': '10-20 epochs',
                'min_delta': '0.001',
                'restore_best_weights': True,
                'benefits': ['Prevents overfitting', 'Saves training time', 'Better generalization']
            },
            'learning_rate_scheduling': {
                'reduce_on_plateau': {
                    'factor': 0.5,
                    'patience': 5,
                    'min_lr': 1e-7,
                    'use_case': 'When validation loss plateaus'
                },
                'exponential_decay': {
                    'initial_lr': 0.001,
                    'decay_rate': 0.96,
                    'decay_steps': 100,
                    'use_case': 'Smooth learning rate reduction'
                }
            },
            'regularization_techniques': {
                'dropout': {
                    'rate': '0.2-0.5',
                    'placement': 'After dense layers',
                    'benefits': 'Prevents overfitting, improves generalization'
                },
                'l1_l2_regularization': {
                    'l1_rate': '0.001-0.01',
                    'l2_rate': '0.001-0.01',
                    'benefits': 'Weight penalty, feature selection (L1), weight decay (L2)'
                },
                'batch_normalization': {
                    'placement': 'Before activation functions',
                    'benefits': 'Faster training, stable gradients, regularization effect'
                }
            }
        },
        'performance_tuning': {
            'memory_optimization': {
                'batch_size_selection': 'Largest size that fits in memory',
                'gradient_accumulation': 'Simulate larger batches with limited memory',
                'model_checkpointing': 'Save memory during long training runs',
                'data_pipeline': 'Efficient data loading and preprocessing'
            },
            'computational_optimization': {
                'gpu_utilization': 'Use GPU acceleration for training',
                'mixed_precision': 'Use float16 for faster training',
                'parallel_processing': 'Multi-GPU training for large models',
                'efficient_architectures': 'Choose architectures appropriate for problem size'
            }
        }
    }


def get_usage_examples() -> Dict[str, str]:
    """Get comprehensive usage examples for common scenarios."""
    return {
        'basic_classification': '''
# Basic Binary Classification Example
from modeling.deep_learning import DeepLearningModel, DeepLearningConfig, ArchitectureType

# Configure the model
config = DeepLearningConfig(
    architecture=ArchitectureType.DENSE,
    hidden_layers=[128, 64, 32],
    dropout_rate=0.3,
    learning_rate=0.001,
    epochs=100,
    batch_size=32,
    validation_split=0.2
)

# Create and train the model
model = DeepLearningModel(config, name="Binary_Classifier")
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Evaluate performance
metrics = model.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"F1-Score: {metrics['f1_score']:.3f}")
''',

        'time_series_forecasting': '''
# LSTM Time Series Forecasting Example
from modeling.deep_learning import DeepLearningModel, DeepLearningConfig, ArchitectureType

# Configure LSTM for time series
config = DeepLearningConfig(
    architecture=ArchitectureType.LSTM,
    lstm_units=[100, 50],
    dropout_rate=0.2,
    sequence_length=60,  # Look back 60 time steps
    learning_rate=0.001,
    epochs=150,
    batch_size=64,
    early_stopping_patience=15
)

# Prepare time series data (X should be 3D: samples x timesteps x features)
model = DeepLearningModel(config, name="LSTM_Forecaster")
model.fit(X_sequences, y_targets)

# Forecast future values
future_predictions = model.predict(X_test_sequences)

# Evaluate forecasting performance
metrics = model.evaluate(X_test_sequences, y_test_targets)
print(f"MAE: {metrics['mae']:.3f}")
print(f"RMSE: {metrics['rmse']:.3f}")
''',

        'hyperparameter_optimization': '''
# Hyperparameter Optimization Example
from modeling.deep_learning import DeepLearningModel, DeepLearningConfig, ArchitectureType

# Define parameter search space
param_grid = {
    'hidden_layers': [[64, 32], [128, 64], [256, 128, 64]],
    'dropout_rate': [0.2, 0.3, 0.4],
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128]
}

# Configure with optimization
config = DeepLearningConfig(
    architecture=ArchitectureType.DENSE,
    optimization_method='grid_search',
    param_grid=param_grid,
    cv_folds=5,
    scoring='f1_score'
)

# Perform optimization
model = DeepLearningModel(config, name="Optimized_Model")
model.fit(X_train, y_train)

# Best parameters found
print(f"Best parameters: {model.best_params_}")
print(f"Best CV score: {model.best_score_:.3f}")

# Use optimized model for prediction
predictions = model.predict(X_test)
''',

        'custom_architecture': '''
# Custom Architecture Example
from modeling.deep_learning import DeepLearningModel, DeepLearningConfig, ArchitectureType
import tensorflow as tf

# Define custom architecture function
def custom_model_builder(input_shape, output_shape, config):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Custom layer sequence
    x = tf.keras.layers.Dense(256, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Branch architecture
    branch1 = tf.keras.layers.Dense(64, activation='relu')(x)
    branch2 = tf.keras.layers.Dense(64, activation='tanh')(x)
    
    # Combine branches
    merged = tf.keras.layers.concatenate([branch1, branch2])
    outputs = tf.keras.layers.Dense(output_shape, activation='softmax')(merged)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Configure custom model
config = DeepLearningConfig(
    architecture=ArchitectureType.CUSTOM,
    custom_model_builder=custom_model_builder,
    learning_rate=0.001,
    epochs=100
)

model = DeepLearningModel(config, name="Custom_Architecture")
model.fit(X_train, y_train)
''',

        'autoencoder_example': '''
# Autoencoder for Dimensionality Reduction
from modeling.deep_learning import DeepLearningModel, DeepLearningConfig, ArchitectureType

# Configure autoencoder
config = DeepLearningConfig(
    architecture=ArchitectureType.AUTOENCODER,
    encoder_layers=[256, 128, 64],
    latent_dim=32,  # Compressed representation size
    decoder_layers=[64, 128, 256],
    learning_rate=0.001,
    epochs=200,
    batch_size=128
)

# Train autoencoder (unsupervised)
model = DeepLearningModel(config, name="Feature_Autoencoder")
model.fit(X_train, X_train)  # Reconstruction task

# Extract features from encoder
encoded_features = model.encode(X_test)
print(f"Original shape: {X_test.shape}")
print(f"Encoded shape: {encoded_features.shape}")

# Reconstruct data
reconstructed = model.predict(X_test)
reconstruction_error = np.mean((X_test - reconstructed) ** 2)
print(f"Reconstruction MSE: {reconstruction_error:.4f}")
'''
    }


def get_performance_benchmarks() -> Dict[str, Any]:
    """Get performance benchmarks and expectations."""
    return {
        'training_performance': {
            'small_models': {
                'description': '1-3 layers, <100K parameters',
                'training_time': '1-5 minutes on CPU, <1 minute on GPU',
                'memory_usage': '<1GB RAM',
                'inference_time': '<1ms per sample',
                'suitable_for': 'Simple tabular data, proof of concepts'
            },
            'medium_models': {
                'description': '4-6 layers, 100K-1M parameters',
                'training_time': '5-30 minutes on CPU, 1-5 minutes on GPU',
                'memory_usage': '1-4GB RAM',
                'inference_time': '1-5ms per sample',
                'suitable_for': 'Complex tabular data, image classification'
            },
            'large_models': {
                'description': '7+ layers, 1M+ parameters',
                'training_time': '30+ minutes on CPU, 5-30 minutes on GPU',
                'memory_usage': '4-16GB RAM',
                'inference_time': '5-50ms per sample',
                'suitable_for': 'Computer vision, complex time series, NLP'
            }
        },
        'accuracy_expectations': {
            'tabular_data': {
                'binary_classification': '85-95% accuracy typical',
                'multi_class': '80-90% accuracy typical',
                'regression': 'R² > 0.8 for good models',
                'factors': 'Data quality, feature engineering, problem complexity'
            },
            'time_series': {
                'forecasting_accuracy': 'MAPE < 10% for good models',
                'trend_prediction': '70-85% directional accuracy',
                'factors': 'Series stationarity, seasonality, noise level'
            },
            'sequence_data': {
                'classification': '75-90% accuracy typical',
                'sequence_to_sequence': 'Task-dependent metrics',
                'factors': 'Sequence length, vocabulary size, context complexity'
            }
        },
        'comparison_with_alternatives': {
            'vs_traditional_ml': {
                'advantages': ['Better with large datasets', 'Automatic feature learning', 'Handle complex patterns'],
                'disadvantages': ['Requires more data', 'Longer training time', 'Less interpretable'],
                'when_to_choose_dl': 'Large datasets (>10K samples), complex patterns, unstructured data'
            },
            'vs_other_dl_frameworks': {
                'tensorflow_keras': 'Industry standard, excellent ecosystem, production ready',
                'pytorch': 'More flexible, research oriented, dynamic graphs',
                'sklearn_mlp': 'Simpler, faster for small problems, limited capabilities'
            }
        }
    }


def generate_info_summary() -> str:
    """Generate a comprehensive summary of the deep learning module."""
    info = get_package_info()
    architectures = get_architecture_breakdown()
    
    summary = f"""
# Deep Learning Module Summary

## Overview
{info['description']}

**Version:** {info['version']}
**Last Updated:** {info['last_updated']}

## Key Capabilities
- **{len(architectures)} Neural Network Architectures** for diverse problem types
- **Automatic Hyperparameter Optimization** with grid/random search
- **Advanced Regularization** including dropout, L1/L2, batch normalization
- **Production-Ready Training** with early stopping, checkpointing, GPU support
- **Comprehensive Evaluation** with 15+ metrics and visualization tools

## Supported Architectures
{', '.join(architectures.keys())}

## Problem Types Supported
- Binary/Multi-class/Multi-label Classification
- Regression (Linear and Non-linear)
- Time Series Forecasting
- Unsupervised Learning (Autoencoders)
- Custom Problem Types

## Performance Features
- **GPU Acceleration:** 10-100x training speedup
- **Memory Optimization:** Efficient batch processing
- **Scalability:** Supports millions of parameters
- **Fast Inference:** <1ms for simple models

## Integration
- ✅ TensorFlow/Keras Backend
- ✅ Scikit-learn Compatible
- ✅ Pandas DataFrame Support
- ✅ BaseModel Interface Compliance
- ✅ Production Deployment Ready

## Quick Start
```python
from modeling.deep_learning import DeepLearningModel, DeepLearningConfig, ArchitectureType

config = DeepLearningConfig(architecture=ArchitectureType.DENSE)
model = DeepLearningModel(config)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

For detailed usage examples and advanced configurations, see the full documentation.
"""
    return summary.strip()


def export_info_json(filename: str = 'deep_learning_info.json') -> None:
    """Export complete module information to JSON file."""
    info_data = {
        'package_info': get_package_info(),
        'architecture_breakdown': get_architecture_breakdown(),
        'optimization_strategies': get_optimization_strategies(),
        'usage_examples': get_usage_examples(),
        'performance_benchmarks': get_performance_benchmarks(),
        'generated_at': datetime.now().isoformat()
    }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(info_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Deep learning module information exported to {filename}")
        print(f"✅ Module information exported to {filename}")
    except Exception as e:
        logger.error(f"Failed to export module information: {e}")
        print(f"❌ Export failed: {e}")


if __name__ == "__main__":
    # Generate and display summary when run directly
    print("🚀 Deep Learning Module Information")
    print("=" * 50)
    print(generate_info_summary())
    print("\n" + "=" * 50)
    
    # Optionally export to JSON
    export_choice = input("\nExport detailed information to JSON? (y/n): ").lower().strip()
    if export_choice in ['y', 'yes']:
        export_info_json()
    
    print("\n📚 Documentation complete!")
