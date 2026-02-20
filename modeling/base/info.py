"""
Base Module Information Module.

This module provides comprehensive information about the foundational base module
capabilities, features, and usage guidelines for standardized interfaces and utilities
used across the modeling system.
"""

import logging
from typing import Dict, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)


def get_package_info() -> Dict[str, Any]:
    """
    Get comprehensive base module information.
    
    Returns:
        Dictionary containing complete module details
    """
    return {
        'module_name': 'Base Modeling Framework',
        'version': '1.0.0',
        'description': 'Foundational framework providing standardized interfaces, abstract base classes, and utilities for the modeling system.',
        'author': 'AI ML Platform Team',
        'license': 'MIT',
        'created_date': '2025-07-27',
        'last_updated': datetime.now().isoformat(),
        'core_components': {
            'base_classes': {
                'file': '__init__.py',
                'lines_of_code': 312,
                'description': 'Abstract base classes and standardized interfaces for all models',
                'key_classes': ['BaseModel', 'ModelMetadata', 'ModelType', 'ProblemType', 'ModelStatus'],
                'features': [
                    'Abstract base class for all models',
                    'Standardized fit/predict interface',
                    'Metadata tracking and versioning',
                    'Status monitoring and error handling',
                    'Type-safe implementations'
                ]
            },
            'training_config': {
                'file': '__init__.py',
                'description': 'Configuration for model training and evaluation',
                'key_classes': ['TrainingConfig', 'DataInfo'],
                'features': [
                    'Train-test split and validation configuration',
                    'Support for cross-validation and hyperparameter tuning',
                    'Early stopping and iteration control',
                    'Dataset statistics and feature information'
                ]
            }
        },
        'supported_features': {
            'model_metadata': {
                'description': 'Comprehensive metadata for machine learning models',
                'key_attributes': ['model_id', 'name', 'description', 'model_type', 'problem_type', 'status', 'version', 'author', 'tags'],
                'usage': 'Track model lifecycle, versioning, and metadata attributes',
                'examples': ['Model versioning', 'Metadata-driven model selection']
            },
            'training_config': {
                'description': 'Configuration for training machine learning models',
                'key_attributes': ['train_test_split', 'validation_split', 'random_state', 'cross_validation_folds', 'hyperparameter_tuning'],
                'usage': 'Define training and evaluation parameters',
                'examples': ['Cross-validation setup', 'Hyperparameter tuning']
            }
        },
        'evaluation_framework': {
            'performance_metrics': {
                'accuracy': {
                    'description': 'Accuracy for classification models',
                    'formula': 'Correct Predictions / Total Predictions',
                    'range': '[0, 1] where 1 is perfect',
                    'best_for': 'Classification evaluation'
                },
                'mse': {
                    'description': 'Mean Squared Error for regression models',
                    'formula': 'Σ(y_true - y_pred)² / n',
                    'range': '[0, ∞] where 0 is perfect',
                    'best_for': 'Regression evaluation'
                },
                'r2': {
                    'description': 'R² Score for regression models',
                    'formula': '1 - (Σ(y_true - y_pred)² / Σ(y_true - mean(y_true))²)',
                    'range': '(-∞, 1] where 1 is perfect',
                    'best_for': 'Explained variance assessment'
                }
            },
            'model_interpretation': {
                'metadata_analysis': 'Analyze model metadata for insights',
                'training_config_analysis': 'Evaluate training configurations for reproducibility',
                'error_analysis': 'Analyze errors and exceptions during model lifecycle'
            }
        }
    }


def export_info_json(filename: str = 'base_info.json') -> None:
    """Export complete module information to JSON file."""
    info_data = {
        'package_info': get_package_info(),
        'generated_at': datetime.now().isoformat()
    }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(info_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Base module information exported to {filename}")
        print(f"✅ Module information exported to {filename}")
    except Exception as e:
        logger.error(f"Failed to export module information: {e}")
        print(f"❌ Export failed: {e}")


if __name__ == "__main__":
    # Generate and display summary when run directly
    print("🎯 Base Module Information")
    print("=" * 60)
    print(json.dumps(get_package_info(), indent=2))
    print("\n" + "=" * 60)
    
    # Optionally export to JSON
    export_choice = input("\nExport detailed information to JSON? (y/n): ").lower().strip()
    if export_choice in ['y', 'yes']:
        export_info_json()
    
    print("\n📚 Documentation complete!")
