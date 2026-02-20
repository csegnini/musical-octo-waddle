"""
Unsupervised Learning Module

This module provides comprehensive unsupervised learning capabilities including:
- Clustering algorithms (K-Means, DBSCAN, Hierarchical, Gaussian Mixture)
- Dimensionality reduction (PCA, t-SNE, UMAP, LLE, Isomap)
- Anomaly detection (Isolation Forest, One-Class SVM, Local Outlier Factor)
- Association rule mining (Apriori, FP-Growth)
- Matrix factorization (NMF, SVD)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering,
    MeanShift, OPTICS, Birch
)
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.decomposition import (
    PCA, TruncatedSVD, FastICA, FactorAnalysis, NMF,
    LatentDirichletAllocation, DictionaryLearning
)
from sklearn.manifold import (
    TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding,
    MDS
)
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score, homogeneity_score,
    completeness_score, v_measure_score
)
from sklearn.model_selection import ParameterGrid
import warnings
warnings.filterwarnings('ignore')

# Optional UMAP import - handled conditionally
try:
    # Use type: ignore to suppress import warnings for optional dependency
    from umap import UMAP  # type: ignore
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    UMAP = None  # Define fallback

# Base classes
from ..base import BaseModel, ModelMetadata, ModelType, ProblemType, ModelStatus
from enum import Enum
from typing import Optional, Dict, List, Tuple, Any, Union
from dataclasses import dataclass
import time
import uuid


class ClusteringAlgorithm(Enum):
    """Supported clustering algorithms."""
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    HIERARCHICAL = "hierarchical"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    SPECTRAL = "spectral"
    MEANSHIFT = "meanshift"
    OPTICS = "optics"
    BIRCH = "birch"


class DimensionalityReductionAlgorithm(Enum):
    """Supported dimensionality reduction algorithms."""
    PCA = "pca"
    TSNE = "tsne"
    UMAP = "umap"
    ISOMAP = "isomap"
    LLE = "lle"
    SPECTRAL_EMBEDDING = "spectral_embedding"
    MDS = "mds"
    ICA = "ica"
    FACTOR_ANALYSIS = "factor_analysis"
    TRUNCATED_SVD = "truncated_svd"
    NMF = "nmf"


class AnomalyDetectionAlgorithm(Enum):
    """Supported anomaly detection algorithms."""
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    LOCAL_OUTLIER_FACTOR = "lof"
    ELLIPTIC_ENVELOPE = "elliptic_envelope"


@dataclass
class ClusteringConfig:
    """Configuration for clustering algorithms."""
    algorithm: ClusteringAlgorithm = ClusteringAlgorithm.KMEANS
    n_clusters: Optional[int] = 3
    random_state: int = 42
    # K-Means specific
    init: str = 'k-means++'
    n_init: int = 10
    max_iter: int = 300
    # DBSCAN specific
    eps: float = 0.5
    min_samples: int = 5
    # Hierarchical specific
    linkage: str = 'ward'
    affinity: str = 'euclidean'
    # Gaussian Mixture specific
    covariance_type: str = 'full'
    # General
    normalize_data: bool = True


@dataclass
class DimensionalityReductionConfig:
    """Configuration for dimensionality reduction algorithms."""
    algorithm: DimensionalityReductionAlgorithm = DimensionalityReductionAlgorithm.PCA
    n_components: int = 2
    random_state: int = 42
    # t-SNE specific
    perplexity: float = 30.0
    learning_rate: float = 200.0
    n_iter: int = 1000
    # UMAP specific
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = 'euclidean'
    # General
    normalize_data: bool = True


@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection algorithms."""
    algorithm: AnomalyDetectionAlgorithm = AnomalyDetectionAlgorithm.ISOLATION_FOREST
    contamination: float = 0.1
    random_state: int = 42
    # Isolation Forest specific
    n_estimators: int = 100
    max_samples: str = 'auto'
    # One-Class SVM specific
    kernel: str = 'rbf'
    gamma: str = 'scale'
    nu: float = 0.05
    # LOF specific
    n_neighbors: int = 20
    algorithm_lof: str = 'auto'
    # General
    normalize_data: bool = True


class ClusteringModel(BaseModel):
    """Advanced clustering model with multiple algorithms."""
    
    def __init__(self, config: ClusteringConfig, name: Optional[str] = None):
        """Initialize clustering model."""
        metadata = ModelMetadata(
            model_id=str(uuid.uuid4()),
            name=name or f"Clustering_{config.algorithm.value}",
            model_type=ModelType.CLUSTERING,
            problem_type=ProblemType.CLUSTERING,
            description=f"Clustering model using {config.algorithm.value} algorithm",
            version="1.0.0",
            status=ModelStatus.UNTRAINED
        )
        super().__init__(metadata)
        
        self.config = config
        self.model = None
        self.scaler = None
        self.labels_ = None
        self.cluster_centers_ = None
        self.silhouette_avg = None
        self.inertia_ = None
        
    def _create_model(self):
        """Create the clustering model based on configuration."""
        config = self.config
        
        if config.algorithm == ClusteringAlgorithm.KMEANS:
            # Ensure n_clusters is not None and init is valid
            n_clusters = config.n_clusters if config.n_clusters is not None else 8  # default value
            # Only allow 'k-means++' or 'random' for init
            init = config.init if config.init in ('k-means++', 'random') else 'k-means++'
            return KMeans(
                n_clusters=n_clusters,
                init=init,
                n_init=config.n_init,
                max_iter=config.max_iter,
                random_state=config.random_state
            )
        
        elif config.algorithm == ClusteringAlgorithm.DBSCAN:
            return DBSCAN(
                eps=config.eps,
                min_samples=config.min_samples
            )
        
        elif config.algorithm == ClusteringAlgorithm.HIERARCHICAL:
            # AgglomerativeClustering does not support 'affinity' when linkage='ward'
            # Only pass 'affinity' if linkage is not 'ward'
            n_clusters = config.n_clusters if config.n_clusters is not None else 2
            # Ensure linkage is one of allowed values
            allowed_linkages = ['ward', 'complete', 'average', 'single']
            linkage = config.linkage if config.linkage in allowed_linkages else 'ward'
            if linkage == 'ward':
                return AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage=linkage  # type: ignore
                )
            else:
                return AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage=linkage  # type: ignore
                )

        elif config.algorithm == ClusteringAlgorithm.GAUSSIAN_MIXTURE:
            n_components = config.n_clusters if config.n_clusters is not None else 1
            covariance_type = config.covariance_type if config.covariance_type in ['full', 'tied', 'diag', 'spherical'] else 'full'
            return GaussianMixture(
                n_components=n_components,
                covariance_type=covariance_type,  # type: ignore
                random_state=config.random_state
            )

        elif config.algorithm == ClusteringAlgorithm.SPECTRAL:
            n_clusters = config.n_clusters if config.n_clusters is not None else 2
            return SpectralClustering(
                n_clusters=n_clusters,
                random_state=config.random_state,
                affinity='nearest_neighbors'
            )

        elif config.algorithm == ClusteringAlgorithm.MEANSHIFT:
            return MeanShift()

        elif config.algorithm == ClusteringAlgorithm.OPTICS:
            return OPTICS(
                min_samples=config.min_samples,
                eps=config.eps
            )

        elif config.algorithm == ClusteringAlgorithm.BIRCH:
            n_clusters = config.n_clusters if config.n_clusters is not None else 3
            return Birch(
                n_clusters=n_clusters
            )

        else:
            raise ValueError(f"Unsupported clustering algorithm: {config.algorithm}")
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.Series]] = None) -> 'ClusteringModel':
        """Fit the clustering model."""
        try:
            self.metadata.status = ModelStatus.TRAINING
            
            # Normalize data if requested
            if self.config.normalize_data:
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = X
            
            # Create and fit model
            self.model = self._create_model()
            
            # Fit and get labels depending on algorithm
            if isinstance(self.model, KMeans) or isinstance(self.model, Birch):
                self.labels_ = self.model.fit_predict(X_scaled)
            elif isinstance(self.model, DBSCAN) or isinstance(self.model, OPTICS) or isinstance(self.model, MeanShift):
                self.labels_ = self.model.fit_predict(X_scaled)
            elif isinstance(self.model, AgglomerativeClustering):
                self.model.fit(X_scaled)
                self.labels_ = self.model.labels_
            elif isinstance(self.model, SpectralClustering):
                self.labels_ = self.model.fit_predict(X_scaled)
            elif isinstance(self.model, GaussianMixture):
                self.model.fit(X_scaled)
                self.labels_ = self.model.predict(X_scaled)
            else:
                self.model.fit(X_scaled)
                if hasattr(self.model, 'labels_'):
                    self.labels_ = self.model.labels_
                else:
                    raise ValueError("Unknown clustering model type for label extraction.")

            # Store cluster centers if available
            if isinstance(self.model, KMeans) or isinstance(self.model, MeanShift):
                self.cluster_centers_ = self.model.cluster_centers_
            elif isinstance(self.model, Birch) and hasattr(self.model, 'subcluster_centers_'):
                self.cluster_centers_ = getattr(self.model, 'subcluster_centers_', None)
            elif isinstance(self.model, GaussianMixture):
                self.cluster_centers_ = self.model.means_
            else:
                self.cluster_centers_ = None

            # Calculate silhouette score
            if len(np.unique(self.labels_)) > 1:
                self.silhouette_avg = silhouette_score(X_scaled, self.labels_)

            # Store inertia for K-means only
            if isinstance(self.model, KMeans):
                self.inertia_ = self.model.inertia_
            else:
                self.inertia_ = None

            self.metadata.status = ModelStatus.TRAINED
            return self
            
        except Exception as e:
            self.metadata.status = ModelStatus.FAILED
            raise e
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict cluster labels for new data."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Scale data if scaler was used during training
        if self.scaler is not None:
            if isinstance(X, pd.DataFrame):
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        if hasattr(self.model, 'predict'):
            return self.model.predict(X_scaled)  # type: ignore
        else:
            # For algorithms without predict method, use fit_predict
            # This is not ideal for new data, but some algorithms don't support prediction
            warnings.warn("This algorithm doesn't support prediction on new data. Refitting on provided data.")
            return self.model.fit_predict(X_scaled)
    
    def get_cluster_evaluation_metrics(self, X: np.ndarray, y_true: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Get clustering evaluation metrics."""
        if self.labels_ is None:
            raise ValueError("Model must be fitted before evaluation")
        
        # Scale data if scaler was used
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        metrics = {}
        
        # Internal validation metrics
        if len(np.unique(self.labels_)) > 1:
            metrics['silhouette_score'] = silhouette_score(X_scaled, self.labels_)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_scaled, self.labels_)
            metrics['davies_bouldin_score'] = davies_bouldin_score(X_scaled, self.labels_)
        
        # External validation metrics (if true labels available)
        if y_true is not None:
            metrics['adjusted_rand_score'] = adjusted_rand_score(y_true, self.labels_)
            metrics['normalized_mutual_info_score'] = normalized_mutual_info_score(y_true, self.labels_)
            metrics['homogeneity_score'] = homogeneity_score(y_true, self.labels_)
            metrics['completeness_score'] = completeness_score(y_true, self.labels_)
            metrics['v_measure_score'] = v_measure_score(y_true, self.labels_)
        
        # Algorithm-specific metrics
        if hasattr(self.model, 'inertia_') and self.model is not None:
            metrics['inertia'] = getattr(self.model, 'inertia_', None)
        
        return metrics
    
    def plot_clusters(self, X: np.ndarray, feature_names: Optional[List[str]] = None, save_path: Optional[str] = None):
        """Plot clustering results."""
        if self.labels_ is None:
            raise ValueError("Model must be fitted before plotting")
        
        # For high-dimensional data, use PCA for visualization
        if X.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            X_plot = pca.fit_transform(X)
            x_label = f"PC1 ({pca.explained_variance_ratio_[0]:.2%} var)"
            y_label = f"PC2 ({pca.explained_variance_ratio_[1]:.2%} var)"
        else:
            pca = None
            X_plot = X
            x_label = feature_names[0] if feature_names else "Feature 1"
            y_label = feature_names[1] if feature_names else "Feature 2"
        
        plt.figure(figsize=(10, 8))
        
        # Plot data points colored by cluster
        unique_labels = np.unique(self.labels_)
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:  # Noise points (DBSCAN, OPTICS)
                plt.scatter(X_plot[self.labels_ == label, 0], 
                          X_plot[self.labels_ == label, 1],
                          c='black', marker='x', s=50, alpha=0.6, label='Noise')
            else:
                plt.scatter(X_plot[self.labels_ == label, 0], 
                          X_plot[self.labels_ == label, 1],
                          c=[color], s=50, alpha=0.6, label=f'Cluster {label}')
        
        # Plot cluster centers if available
        if self.cluster_centers_ is not None:
            centers = np.array(self.cluster_centers_)
            if X.shape[1] > 2 and pca is not None:
                centers_plot = pca.transform(centers)
            else:
                centers_plot = centers
            
            plt.scatter(centers_plot[:, 0], centers_plot[:, 1],
                       c='red', marker='*', s=200, alpha=0.8, 
                       edgecolors='black', linewidth=2, label='Centers')
        
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f'{self.config.algorithm.value.title()} Clustering Results\n'
                 f'Silhouette Score: {self.silhouette_avg:.3f}' if self.silhouette_avg else '')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class DimensionalityReductionModel(BaseModel):
    """Advanced dimensionality reduction model."""
    
    def __init__(self, config: DimensionalityReductionConfig, name: Optional[str] = None):
        """Initialize dimensionality reduction model."""
        metadata = ModelMetadata(
            model_id=str(uuid.uuid4()),
            name=name or f"DimReduction_{config.algorithm.value}",
            model_type=ModelType.UNSUPERVISED if hasattr(ModelType, "UNSUPERVISED") else ModelType.CLUSTERING,
            problem_type=ProblemType.DIMENSIONALITY_REDUCTION if hasattr(ProblemType, "DIMENSIONALITY_REDUCTION") else ProblemType.CLUSTERING,
            description=f"Dimensionality reduction model using {config.algorithm.value} algorithm",
            version="1.0.0",
            status=ModelStatus.UNTRAINED
        )
        super().__init__(metadata)
        
        self.config = config
        self.model = None
        self.scaler = None
        self.explained_variance_ratio_ = None
        self.embedding_ = None
        
    def _create_model(self):
        """Create the dimensionality reduction model."""
        config = self.config
        
        if config.algorithm == DimensionalityReductionAlgorithm.PCA:
            return PCA(
                n_components=config.n_components,
                random_state=config.random_state
            )
        
        elif config.algorithm == DimensionalityReductionAlgorithm.TSNE:
            return TSNE(
                n_components=config.n_components,
                perplexity=config.perplexity,
                learning_rate=config.learning_rate,
                n_iter=config.n_iter,
                random_state=config.random_state
            )
        
        elif config.algorithm == DimensionalityReductionAlgorithm.UMAP:
            if not UMAP_AVAILABLE:
                raise ImportError("UMAP not available. Install with: pip install umap-learn")
            # Use the globally imported UMAP if available
            if 'UMAP' in globals():
                return globals()['UMAP'](
                    n_components=config.n_components,
                    n_neighbors=config.n_neighbors,
                    min_dist=config.min_dist,
                    metric=config.metric,
                    random_state=config.random_state
                )
            else:
                raise ImportError("UMAP not available. Install with: pip install umap-learn")
        
        elif config.algorithm == DimensionalityReductionAlgorithm.ISOMAP:
            return Isomap(
                n_components=config.n_components,
                n_neighbors=config.n_neighbors
            )
        
        elif config.algorithm == DimensionalityReductionAlgorithm.LLE:
            return LocallyLinearEmbedding(
                n_components=config.n_components,
                n_neighbors=config.n_neighbors,
                random_state=config.random_state
            )
        
        elif config.algorithm == DimensionalityReductionAlgorithm.SPECTRAL_EMBEDDING:
            return SpectralEmbedding(
                n_components=config.n_components,
                random_state=config.random_state
            )
        
        elif config.algorithm == DimensionalityReductionAlgorithm.MDS:
            return MDS(
                n_components=config.n_components,
                random_state=config.random_state
            )
        
        elif config.algorithm == DimensionalityReductionAlgorithm.ICA:
            return FastICA(
                n_components=config.n_components,
                random_state=config.random_state
            )
        
        elif config.algorithm == DimensionalityReductionAlgorithm.FACTOR_ANALYSIS:
            return FactorAnalysis(
                n_components=config.n_components,
                random_state=config.random_state
            )
        
        elif config.algorithm == DimensionalityReductionAlgorithm.TRUNCATED_SVD:
            return TruncatedSVD(
                n_components=config.n_components,
                random_state=config.random_state
            )
        
        elif config.algorithm == DimensionalityReductionAlgorithm.NMF:
            return NMF(
                n_components=config.n_components,
                random_state=config.random_state
            )
        
        else:
            raise ValueError(f"Unsupported dimensionality reduction algorithm: {config.algorithm}")
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.Series]] = None) -> 'DimensionalityReductionModel':
        """Fit the dimensionality reduction model."""
        try:
            self.metadata.status = ModelStatus.TRAINING
            
            # Normalize data if requested
            if self.config.normalize_data:
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = X
            
            # Create and fit model
            self.model = self._create_model()
            
            if hasattr(self.model, 'fit_transform'):
                self.embedding_ = self.model.fit_transform(X_scaled)
            else:
                self.model.fit(X_scaled)
                if hasattr(self.model, 'transform'):
                    self.embedding_ = self.model.transform(X_scaled)  # type: ignore
                else:
                    # For algorithms without transform method, store the fitted result
                    self.embedding_ = self.model.fit_transform(X_scaled)
            
            # Store explained variance ratio if available and not None
            if hasattr(self.model, 'explained_variance_ratio_'):
                self.explained_variance_ratio_ = getattr(self.model, 'explained_variance_ratio_', None)
            else:
                self.explained_variance_ratio_ = None
            
            self.metadata.status = ModelStatus.TRAINED
            return self
            
        except Exception as e:
            self.metadata.status = ModelStatus.FAILED
            raise e
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Transform new data to reduced dimensions."""
        if self.model is None:
            raise ValueError("Model must be fitted before transformation")
        
        # Scale data if scaler was used during training
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        if hasattr(self.model, 'transform'):
            return self.model.transform(X_scaled)  # type: ignore
        else:
            # For algorithms without transform method, refit (not ideal)
            warnings.warn("This algorithm doesn't support transformation of new data. Refitting on provided data.")
            return self.model.fit_transform(X_scaled)
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """For compatibility with BaseModel - delegates to transform."""
        return self.transform(X)
    
    def plot_embedding(self, y: Optional[np.ndarray] = None, feature_names: Optional[List[str]] = None, save_path: Optional[str] = None):
        """Plot the dimensionality reduction results."""
        if self.embedding_ is None:
            raise ValueError("Model must be fitted before plotting")
        
        plt.figure(figsize=(10, 8))
        
        if self.config.n_components == 2:
            if y is not None:
                # Color by target labels
                unique_labels = np.unique(y)
                colors = plt.cm.get_cmap('Set3')(np.linspace(0, 1, len(unique_labels)))
                
                for label, color in zip(unique_labels, colors):
                    mask = y == label
                    plt.scatter(self.embedding_[mask, 0], self.embedding_[mask, 1],
                              c=[color], s=50, alpha=0.6, label=f'Class {label}')
                plt.legend()
            else:
                plt.scatter(self.embedding_[:, 0], self.embedding_[:, 1], 
                          s=50, alpha=0.6, c='blue')
            
            plt.xlabel(f'Component 1')
            plt.ylabel(f'Component 2')
            
        elif self.config.n_components == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            if y is not None:
                unique_labels = np.unique(y)
                cmap = plt.get_cmap('tab10')
                colors = [cmap(i) for i in range(len(unique_labels))]
                
                for label, color in zip(unique_labels, colors):
                    mask = y == label
                    ax.scatter(self.embedding_[mask, 0], self.embedding_[mask, 1], 
                               self.embedding_[mask, 2], color=color, alpha=0.6, 
                               label=f'Class {label}')
                ax.legend()
            else:
                ax.scatter(self.embedding_[:, 0], self.embedding_[:, 1], 
                           self.embedding_[:, 2], color='blue', alpha=0.6)
            
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            try:
                ax.set_zlabel('Component 3')  # type: ignore
            except (AttributeError, TypeError):
                pass  # Some versions may not have set_zlabel or may not support it
        
        title = f'{self.config.algorithm.value.upper()} Embedding'
        if self.explained_variance_ratio_ is not None:
            var_explained = sum(self.explained_variance_ratio_[:self.config.n_components])
            title += f'\nExplained Variance: {var_explained:.2%}'
        
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class AnomalyDetectionModel(BaseModel):
    """Advanced anomaly detection model."""
    
    def __init__(self, config: AnomalyDetectionConfig, name: Optional[str] = None):
        """Initialize anomaly detection model."""
        metadata = ModelMetadata(
            model_id=str(uuid.uuid4()),
            name=name or f"AnomalyDetection_{config.algorithm.value}",
            model_type=ModelType.UNSUPERVISED,
            problem_type=ProblemType.ANOMALY_DETECTION,
            description=f"Anomaly detection model using {config.algorithm.value} algorithm",
            version="1.0.0",
            status=ModelStatus.UNTRAINED
        )
        super().__init__(metadata)
        
        self.config = config
        self.model = None
        self.scaler = None
        self.anomaly_scores_ = None
        
    def _create_model(self):
        """Create the anomaly detection model."""
        config = self.config
        
        if config.algorithm == AnomalyDetectionAlgorithm.ISOLATION_FOREST:
            # max_samples can be float or 'auto'
            max_samples = config.max_samples
            if isinstance(max_samples, str):
                if max_samples == 'auto':
                    max_samples = 'auto'
                else:
                    try:
                        max_samples = float(max_samples)
                    except ValueError:
                        max_samples = 'auto'
            return IsolationForest(
                contamination=config.contamination,
                n_estimators=config.n_estimators,
                max_samples=max_samples,
                random_state=config.random_state
            )
        
        elif config.algorithm == AnomalyDetectionAlgorithm.ONE_CLASS_SVM:
            # kernel must be one of allowed literals
            allowed_kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
            kernel = config.kernel if config.kernel in allowed_kernels else 'rbf'
            # gamma can be float or 'scale'/'auto'
            gamma = config.gamma
            if isinstance(gamma, str):
                if gamma in ['scale', 'auto']:
                    gamma = gamma
                else:
                    try:
                        gamma = float(gamma)
                    except ValueError:
                        gamma = 'scale'
            return OneClassSVM(
                kernel=kernel,  # type: ignore
                gamma=gamma,    # type: ignore
                nu=config.nu
            )
        
        elif config.algorithm == AnomalyDetectionAlgorithm.LOCAL_OUTLIER_FACTOR:
            allowed_algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
            algorithm_lof = config.algorithm_lof if config.algorithm_lof in allowed_algorithms else 'auto'
            return LocalOutlierFactor(
                n_neighbors=config.n_neighbors,
                algorithm=algorithm_lof,  # type: ignore
                contamination=config.contamination
            )
        
        elif config.algorithm == AnomalyDetectionAlgorithm.ELLIPTIC_ENVELOPE:
            return EllipticEnvelope(
                contamination=config.contamination,
                random_state=config.random_state
            )
        
        else:
            raise ValueError(f"Unsupported anomaly detection algorithm: {config.algorithm}")
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.Series]] = None) -> 'AnomalyDetectionModel':
        """Fit the anomaly detection model."""
        try:
            self.metadata.status = ModelStatus.TRAINING
            
            # Normalize data if requested
            if self.config.normalize_data:
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = X
            
            # Create and fit model
            self.model = self._create_model()
            
            if hasattr(self.model, 'fit_predict'):
                # For LOF which doesn't have separate fit and predict
                self.anomaly_scores_ = self.model.fit_predict(X_scaled)
            else:
                self.model.fit(X_scaled)
                self.anomaly_scores_ = self.model.predict(X_scaled)
            
            self.metadata.status = ModelStatus.TRAINED
            return self
            
        except Exception as e:
            self.metadata.status = ModelStatus.FAILED
            raise e
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict anomalies for new data."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Scale data if scaler was used during training
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict(X_scaled)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores for new data."""
        if self.model is None:
            raise ValueError("Model must be fitted before scoring")
        
        # Scale data if scaler was used during training
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        if hasattr(self.model, 'decision_function'):
            return self.model.decision_function(X_scaled)
        elif hasattr(self.model, 'score_samples'):
            return np.asarray(self.model.score_samples(X_scaled))
        else:
            return self.model.predict(X_scaled)


# Factory functions for easy model creation
def create_clustering_model(
    algorithm: ClusteringAlgorithm = ClusteringAlgorithm.KMEANS,
    n_clusters: int = 3,
    **kwargs
) -> ClusteringModel:
    """Create a clustering model with specified configuration."""
    config = ClusteringConfig(
        algorithm=algorithm,
        n_clusters=n_clusters,
        **kwargs
    )
    return ClusteringModel(config)


def create_dimensionality_reduction_model(
    algorithm: DimensionalityReductionAlgorithm = DimensionalityReductionAlgorithm.PCA,
    n_components: int = 2,
    **kwargs
) -> DimensionalityReductionModel:
    """Create a dimensionality reduction model with specified configuration."""
    config = DimensionalityReductionConfig(
        algorithm=algorithm,
        n_components=n_components,
        **kwargs
    )
    return DimensionalityReductionModel(config)


def create_anomaly_detection_model(
    algorithm: AnomalyDetectionAlgorithm = AnomalyDetectionAlgorithm.ISOLATION_FOREST,
    contamination: float = 0.1,
    **kwargs
) -> AnomalyDetectionModel:
    """Create an anomaly detection model with specified configuration."""
    config = AnomalyDetectionConfig(
        algorithm=algorithm,
        contamination=contamination,
        **kwargs
    )
    return AnomalyDetectionModel(config)


# Utility functions
def find_optimal_clusters(X: np.ndarray, max_clusters: int = 10, algorithm: str = 'kmeans') -> Dict[str, Any]:
    """Find optimal number of clusters using multiple metrics."""
    silhouette_scores = []
    inertias = []
    calinski_scores = []
    davies_bouldin_scores = []
    
    cluster_range = range(2, max_clusters + 1)
    
    for n_clusters in cluster_range:
        if algorithm == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42)
            labels = model.fit_predict(X)
            inertias.append(model.inertia_)
        elif algorithm == 'gaussian_mixture':
            model = GaussianMixture(n_components=n_clusters, random_state=42)
            labels = model.fit_predict(X)
        else:
            raise ValueError(f"Unsupported algorithm for optimization: {algorithm}")
        
        silhouette_scores.append(silhouette_score(X, labels))
        calinski_scores.append(calinski_harabasz_score(X, labels))
        davies_bouldin_scores.append(davies_bouldin_score(X, labels))
    
    # Find optimal based on different metrics
    optimal_silhouette = cluster_range[np.argmax(silhouette_scores)]
    optimal_calinski = cluster_range[np.argmax(calinski_scores)]
    optimal_davies_bouldin = cluster_range[np.argmin(davies_bouldin_scores)]
    
    results = {
        'cluster_range': list(cluster_range),
        'silhouette_scores': silhouette_scores,
        'calinski_scores': calinski_scores,
        'davies_bouldin_scores': davies_bouldin_scores,
        'optimal_silhouette': optimal_silhouette,
        'optimal_calinski': optimal_calinski,
        'optimal_davies_bouldin': optimal_davies_bouldin
    }
    
    if algorithm == 'kmeans':
        results['inertias'] = inertias
        # Elbow method
        if len(inertias) > 2:
            # Simple elbow detection using the "elbow" heuristic
            diffs = np.diff(inertias)
            diffs2 = np.diff(diffs)
            elbow_idx = np.argmax(diffs2) + 2  # +2 because of double diff
            results['optimal_elbow'] = cluster_range[min(int(elbow_idx), len(cluster_range) - 1)]
    
    return results


def plot_cluster_optimization(optimization_results: Dict[str, Any], save_path: Optional[str] = None):
    """Plot cluster optimization results."""
    cluster_range = optimization_results['cluster_range']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Silhouette score
    axes[0, 0].plot(cluster_range, optimization_results['silhouette_scores'], 'bo-')
    axes[0, 0].set_xlabel('Number of Clusters')
    axes[0, 0].set_ylabel('Silhouette Score')
    axes[0, 0].set_title('Silhouette Analysis')
    axes[0, 0].grid(True)
    
    # Calinski-Harabasz score
    axes[0, 1].plot(cluster_range, optimization_results['calinski_scores'], 'ro-')
    axes[0, 1].set_xlabel('Number of Clusters')
    axes[0, 1].set_ylabel('Calinski-Harabasz Score')
    axes[0, 1].set_title('Calinski-Harabasz Analysis')
    axes[0, 1].grid(True)
    
    # Davies-Bouldin score
    axes[1, 0].plot(cluster_range, optimization_results['davies_bouldin_scores'], 'go-')
    axes[1, 0].set_xlabel('Number of Clusters')
    axes[1, 0].set_ylabel('Davies-Bouldin Score')
    axes[1, 0].set_title('Davies-Bouldin Analysis (Lower is Better)')
    axes[1, 0].grid(True)
    
    # Elbow method (if available)
    if 'inertias' in optimization_results:
        axes[1, 1].plot(cluster_range, optimization_results['inertias'], 'mo-')
        axes[1, 1].set_xlabel('Number of Clusters')
        axes[1, 1].set_ylabel('Inertia')
        axes[1, 1].set_title('Elbow Method')
        axes[1, 1].grid(True)
        
        if 'optimal_elbow' in optimization_results:
            axes[1, 1].axvline(x=optimization_results['optimal_elbow'], 
                             color='red', linestyle='--', 
                             label=f"Elbow at k={optimization_results['optimal_elbow']}")
            axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, 'Elbow Method\nNot Available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# Export all classes and functions
__all__ = [
    # Main classes
    'ClusteringModel', 'DimensionalityReductionModel', 'AnomalyDetectionModel',
    
    # Configuration classes
    'ClusteringConfig', 'DimensionalityReductionConfig', 'AnomalyDetectionConfig',
    
    # Enums
    'ClusteringAlgorithm', 'DimensionalityReductionAlgorithm', 'AnomalyDetectionAlgorithm',
    
    # Factory functions
    'create_clustering_model', 'create_dimensionality_reduction_model', 'create_anomaly_detection_model',
    
    # Utility functions
    'find_optimal_clusters', 'plot_cluster_optimization'
]
