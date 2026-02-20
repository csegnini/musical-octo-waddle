"""
Unsupervised Learning Module Information Module.

This module provides comprehensive information about the unsupervised learning module
capabilities, features, and usage guidelines for clustering, dimensionality reduction,
anomaly detection, and other unsupervised learning techniques.
"""

import logging
from typing import Dict, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)


def get_package_info() -> Dict[str, Any]:
    """
    Get comprehensive unsupervised learning module information.
    
    Returns:
        Dictionary containing complete module details
    """
    return {
        'module_name': 'Advanced Unsupervised Learning Framework',
        'version': '1.0.0',
        'description': 'Comprehensive unsupervised learning framework with clustering, dimensionality reduction, anomaly detection, and association rule mining capabilities.',
        'author': 'AI ML Platform Team',
        'license': 'MIT',
        'created_date': '2025-07-27',
        'last_updated': datetime.now().isoformat(),
        'core_components': {
            'unsupervised_models': {
                'file': '__init__.py',
                'lines_of_code': 968,
                'description': 'Advanced unsupervised learning models with comprehensive evaluation and interpretation capabilities',
                'key_classes': ['ClusteringModel', 'DimensionalityReductionConfig', 'AnomalyDetectionConfig'],
                'features': [
                    'Clustering algorithms (K-Means, DBSCAN, Hierarchical, Gaussian Mixture)',
                    'Dimensionality reduction (PCA, t-SNE, UMAP, LLE, Isomap)',
                    'Anomaly detection (Isolation Forest, One-Class SVM, Local Outlier Factor)',
                    'Association rule mining (Apriori, FP-Growth)',
                    'Matrix factorization (NMF, SVD)',
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
            'clustering': {
                'kmeans': {
                    'description': 'K-Means clustering for partitioning data into k clusters',
                    'class_name': 'ClusteringModel',
                    'algorithm_type': 'Clustering',
                    'strengths': ['Fast and efficient for large datasets', 'Easy to implement'],
                    'weaknesses': ['Sensitive to initialization', 'Assumes spherical clusters'],
                    'best_use_cases': ['Customer segmentation', 'Image compression'],
                    'hyperparameters': {
                        'n_clusters': 'int (number of clusters)',
                        'init': 'str (initialization method)',
                        'max_iter': 'int (maximum iterations)',
                        'random_state': 'int (random seed)'
                    },
                    'complexity': 'O(n × k × i) where n=points, k=clusters, i=iterations',
                    'output_types': ['cluster_labels', 'centroids']
                },
                'dbscan': {
                    'description': 'Density-Based Spatial Clustering of Applications with Noise',
                    'class_name': 'ClusteringModel',
                    'algorithm_type': 'Clustering',
                    'strengths': ['Identifies clusters of arbitrary shape', 'Robust to noise'],
                    'weaknesses': ['Sensitive to parameter selection', 'Not suitable for high-dimensional data'],
                    'best_use_cases': ['Geospatial data analysis', 'Anomaly detection'],
                    'hyperparameters': {
                        'eps': 'float (maximum distance between points in a cluster)',
                        'min_samples': 'int (minimum points to form a cluster)'
                    },
                    'complexity': 'O(n log n) for low-dimensional data',
                    'output_types': ['cluster_labels']
                }
            },
            'dimensionality_reduction': {
                'pca': {
                    'description': 'Principal Component Analysis for reducing data dimensionality',
                    'class_name': 'DimensionalityReductionConfig',
                    'algorithm_type': 'Dimensionality Reduction',
                    'strengths': ['Reduces dimensionality while preserving variance', 'Improves computational efficiency'],
                    'weaknesses': ['Linear method', 'Sensitive to scaling'],
                    'best_use_cases': ['Feature extraction', 'Data visualization'],
                    'hyperparameters': {
                        'n_components': 'int (number of components)',
                        'random_state': 'int (random seed)'
                    },
                    'complexity': 'O(n × p²) where n=samples, p=features',
                    'output_types': ['transformed_data', 'explained_variance']
                },
                'tsne': {
                    'description': 't-Distributed Stochastic Neighbor Embedding for non-linear dimensionality reduction',
                    'class_name': 'DimensionalityReductionConfig',
                    'algorithm_type': 'Dimensionality Reduction',
                    'strengths': ['Captures non-linear relationships', 'Effective for visualization'],
                    'weaknesses': ['High computational cost', 'Non-deterministic'],
                    'best_use_cases': ['Data visualization', 'Exploratory data analysis'],
                    'hyperparameters': {
                        'perplexity': 'float (balance between local/global structure)',
                        'learning_rate': 'float (step size)',
                        'n_iter': 'int (number of iterations)'
                    },
                    'complexity': 'O(n²) where n=samples',
                    'output_types': ['transformed_data']
                }
            }
        },
        'evaluation_framework': {
            'performance_metrics': {
                'silhouette_score': {
                    'description': 'Silhouette Coefficient for cluster quality',
                    'formula': '(b - a) / max(a, b)',
                    'range': '[-1, 1] where 1 is perfect',
                    'best_for': 'Clustering evaluation'
                },
                'calinski_harabasz_score': {
                    'description': 'Variance ratio criterion for cluster quality',
                    'formula': '(SSB / (k - 1)) / (SSW / (n - k))',
                    'range': '[0, ∞] where higher is better',
                    'best_for': 'Clustering evaluation'
                },
                'davies_bouldin_score': {
                    'description': 'Cluster separation measure',
                    'formula': 'Σ(max(Rij)) / k',
                    'range': '[0, ∞] where lower is better',
                    'best_for': 'Clustering evaluation'
                }
            },
            'model_interpretation': {
                'cluster_centroids': 'Centroids for K-Means clustering',
                'explained_variance': 'Variance explained by PCA components',
                'residual_analysis': 'Analyze residuals to assess model fit'
            }
        }
    }


def export_info_json(filename: str = 'unsupervised_info.json') -> None:
    """Export complete module information to JSON file."""
    info_data = {
        'package_info': get_package_info(),
        'generated_at': datetime.now().isoformat()
    }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(info_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Unsupervised learning module information exported to {filename}")
        print(f"✅ Module information exported to {filename}")
    except Exception as e:
        logger.error(f"Failed to export module information: {e}")
        print(f"❌ Export failed: {e}")


if __name__ == "__main__":
    # Generate and display summary when run directly
    print("🎯 Unsupervised Learning Module Information")
    print("=" * 60)
    print(json.dumps(get_package_info(), indent=2))
    print("\n" + "=" * 60)
    
    # Optionally export to JSON
    export_choice = input("\nExport detailed information to JSON? (y/n): ").lower().strip()
    if export_choice in ['y', 'yes']:
        export_info_json()
    
    print("\n📚 Documentation complete!")
