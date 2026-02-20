"""
Base Modeling Package

This module provides the foundational classes and interfaces for the modeling system.
It includes abstract base classes, common utilities, and standardized interfaces
that all modeling components inherit from.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of machine learning models."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    NEURAL_NETWORK = "neural_network"
    TIME_SERIES = "time_series"
    ENSEMBLE = "ensemble"
    UNSUPERVISED = "unsupervised"


class ModelStatus(Enum):
    """Status of a model."""
    UNTRAINED = "untrained"
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = 'failed'


class ProblemType(Enum):
    """Types of machine learning problems."""
    LINEAR_REGRESSION = "linear_regression"
    POLYNOMIAL_REGRESSION = "polynomial_regression"
    LOGISTIC_REGRESSION = "logistic_regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    CLUSTERING = "clustering"
    TIME_SERIES_FORECASTING = "time_series_forecasting"
    ANOMALY_DETECTION = "anomaly_detection"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"


@dataclass
class ModelMetadata:
    """Metadata for a machine learning model."""
    model_id: str
    name: str
    description: str
    model_type: ModelType
    problem_type: ProblemType
    status: ModelStatus = ModelStatus.UNTRAINED
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    author: str = ""
    tags: List[str] = field(default_factory=list)
    features: List[str] = field(default_factory=list)
    target: Optional[str] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    training_time: Optional[float] = None
    data_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    train_test_split: float = 0.8
    validation_split: float = 0.2
    random_state: Optional[int] = 42
    shuffle: bool = True
    stratify: bool = False
    cross_validation_folds: int = 5
    hyperparameter_tuning: bool = False
    early_stopping: bool = False
    max_iterations: int = 1000
    tolerance: float = 1e-6
    verbose: bool = False


@dataclass
class DataInfo:
    """Information about the dataset."""
    n_samples: int
    n_features: int
    feature_names: List[str]
    target_name: Optional[str] = None
    data_types: Dict[str, str] = field(default_factory=dict)
    missing_values: Dict[str, int] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)


class BaseModel(ABC):
    """
    Abstract base class for all machine learning models.
    
    This class defines the standard interface that all models must implement,
    ensuring consistency across different model types and algorithms.
    """
    
    def __init__(self, metadata: ModelMetadata, config: Optional[TrainingConfig] = None):
        """
        Initialize the base model.
        
        Args:
            metadata: Model metadata
            config: Training configuration
        """
        self.metadata = metadata
        self.config = config or TrainingConfig()
        self.is_fitted = False
        self.training_history = []
        self.feature_names = None
        self.target_name = None
        self._model = None
        
        logger.info(f"Initialized {self.metadata.name} model")
    
    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'BaseModel':
        """
        Train the model on the given data.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Fitted model instance
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions
        """
        pass
    
    def fit_predict(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Fit the model and make predictions on the same data.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Predictions on training data
        """
        return self.fit(X, y).predict(X)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters.
        
        Args:
            deep: If True, will return parameters for this estimator and 
                  sub-estimators (for compatibility with sklearn)
                  
        Returns:
            Model parameters
        """
        return self.metadata.hyperparameters.copy()
    
    def set_params(self, **params) -> 'BaseModel':
        """Set model parameters."""
        self.metadata.hyperparameters.update(params)
        return self
    
    def get_metadata(self) -> ModelMetadata:
        """Get model metadata."""
        return self.metadata
    
    def update_status(self, status: ModelStatus) -> None:
        """Update model status."""
        self.metadata.status = status
        self.metadata.updated_at = datetime.utcnow()
        logger.info(f"Model {self.metadata.name} status updated to {status.value}")
    
    def add_metric(self, metric_name: str, value: float) -> None:
        """Add a performance metric."""
        self.metadata.metrics[metric_name] = value
        logger.info(f"Added metric {metric_name}: {value}")
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available."""
        if self._model is not None and hasattr(self._model, 'feature_importances_'):
            if self.feature_names:
                return dict(zip(self.feature_names, self._model.feature_importances_))
            else:
                return {f"feature_{i}": imp for i, imp in enumerate(self._model.feature_importances_)}
        return None
    
    def _validate_data(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.Series]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Validate and prepare input data."""
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        elif isinstance(X, np.ndarray):
            if self.feature_names is None:
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        if y is not None:
            if isinstance(y, pd.Series):
                self.target_name = str(y.name) if y.name is not None else "target"
                y = np.asarray(y.values)
            elif isinstance(y, np.ndarray):
                if self.target_name is None:
                    self.target_name = "target"
        
        # Basic validation
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        
        # Ensure y is numpy array or None
        y_array = np.array(y) if y is not None else None
        
        return X, y_array
    
    def _get_data_info(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> DataInfo:
        """Extract information about the dataset."""
        feature_names = self.feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        # Basic statistics
        stats = {
            'mean': np.mean(X, axis=0).tolist(),
            'std': np.std(X, axis=0).tolist(),
            'min': np.min(X, axis=0).tolist(),
            'max': np.max(X, axis=0).tolist()
        }
        
        return DataInfo(
            n_samples=X.shape[0],
            n_features=X.shape[1],
            feature_names=feature_names,
            target_name=str(self.target_name) if self.target_name is not None else None,
            statistics=stats
        )
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(name='{self.metadata.name}', status='{self.metadata.status.value}')"


class BasePreprocessor(ABC):
    """
    Abstract base class for data preprocessors.
    """
    
    def __init__(self, name: str):
        """Initialize preprocessor."""
        self.name = name
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.Series]] = None) -> 'BasePreprocessor':
        """Fit the preprocessor."""
        pass
    
    @abstractmethod
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Transform the data."""
        pass
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.Series]] = None) -> Union[np.ndarray, pd.DataFrame]:
        """Fit and transform the data."""
        return self.fit(X, y).transform(X)


class BaseEvaluator(ABC):
    """
    Abstract base class for model evaluators.
    """
    
    def __init__(self, metrics: List[str]):
        """Initialize evaluator."""
        self.metrics = metrics
    
    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate predictions against true values."""
        pass
    
    @abstractmethod
    def generate_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Generate a comprehensive evaluation report."""
        pass
