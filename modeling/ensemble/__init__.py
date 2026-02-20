"""
Ensemble Methods Package

This module provides comprehensive ensemble learning capabilities including:
- Voting ensembles (hard and soft voting)
- Bagging and Random Subspace methods
- Boosting algorithms (AdaBoost, Gradient Boosting)
- Stacking and multi-level ensembles
- Blending techniques
- Custom ensemble builders

All ensemble methods integrate seamlessly with the base modeling framework
and support both regression and classification tasks.
"""

import uuid
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum

# Sklearn ensemble methods
from sklearn.ensemble import (
    VotingClassifier, VotingRegressor,
    BaggingClassifier, BaggingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report

# Import base classes
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from base import BaseModel, ModelMetadata, ModelType, ModelStatus, ProblemType, TrainingConfig


class EnsembleType(Enum):
    """Types of ensemble methods."""
    VOTING = "voting"
    BAGGING = "bagging"
    BOOSTING = "boosting"
    STACKING = "stacking"
    BLENDING = "blending"
    CUSTOM = "custom"


@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods."""
    ensemble_type: EnsembleType
    base_estimators: List[Any] = field(default_factory=list)
    n_estimators: int = 10
    voting: str = "hard"  # "hard" or "soft"
    bootstrap: bool = True
    max_samples: Union[int, float] = 1.0
    max_features: Union[int, float] = 1.0
    random_state: Optional[int] = 42
    n_jobs: Optional[int] = -1
    use_cross_validation: bool = True
    cv_folds: int = 5
    meta_learner: Optional[Any] = None
    stack_method: str = "auto"  # "auto", "predict", "predict_proba"


class BaseEnsembleModel(BaseModel):
    """
    Base class for ensemble models.
    
    Provides common functionality for all ensemble methods including
    base estimator management, cross-validation, and evaluation.
    """
    
    def __init__(self, 
                 metadata: ModelMetadata,
                 ensemble_config: EnsembleConfig,
                 config: Optional[TrainingConfig] = None):
        """
        Initialize ensemble model.
        
        Args:
            metadata: Model metadata
            ensemble_config: Ensemble-specific configuration
            config: General training configuration
        """
        super().__init__(metadata, config)
        self.ensemble_config = ensemble_config
        self.base_models = []
        self.ensemble_model = None
        self.base_predictions = None
        self.meta_features = None
        
        # Store original estimators
        self.base_estimators = ensemble_config.base_estimators.copy()
        
    def _prepare_base_estimators(self) -> List[Any]:
        """Prepare base estimators for ensemble."""
        if not self.base_estimators:
            # Create default base estimators based on problem type
            return self._get_default_estimators()
        
        # Clone estimators to avoid issues with multiple fits
        return [clone(estimator) for estimator in self.base_estimators]
    
    def _get_default_estimators(self) -> List[Any]:
        """Get default base estimators based on problem type."""
        if self.metadata.model_type == ModelType.CLASSIFICATION:
            return [
                LogisticRegression(max_iter=1000, random_state=self.ensemble_config.random_state),
                DecisionTreeClassifier(random_state=self.ensemble_config.random_state),
                SVC(probability=True, random_state=self.ensemble_config.random_state),
                KNeighborsClassifier(),
                GaussianNB()
            ]
        else:  # Regression
            return [
                LinearRegression(),
                DecisionTreeRegressor(random_state=self.ensemble_config.random_state),
                SVR(),
                KNeighborsRegressor()
            ]
    
    def _evaluate_base_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate individual base models using cross-validation."""
        base_scores = {}
        
        if self.ensemble_config.use_cross_validation:
            # Choose appropriate cross-validation strategy
            if self.metadata.model_type == ModelType.CLASSIFICATION:
                cv = StratifiedKFold(n_splits=self.ensemble_config.cv_folds, 
                                   shuffle=True, 
                                   random_state=self.ensemble_config.random_state)
                scoring = 'accuracy'
            else:
                cv = KFold(n_splits=self.ensemble_config.cv_folds, 
                          shuffle=True, 
                          random_state=self.ensemble_config.random_state)
                scoring = 'neg_mean_squared_error'
            
            for i, estimator in enumerate(self.base_estimators):
                try:
                    scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring)
                    base_scores[f'base_model_{i}'] = np.mean(scores)
                except Exception as e:
                    print(f"Warning: Could not evaluate base model {i}: {e}")
                    base_scores[f'base_model_{i}'] = 0.0
        
        return base_scores
    
    def get_base_model_performance(self) -> Dict[str, float]:
        """Get performance metrics for base models."""
        return getattr(self, '_base_scores', {})


class VotingEnsembleModel(BaseEnsembleModel):
    """
    Voting ensemble model for both classification and regression.
    
    Combines predictions from multiple base models using either:
    - Hard voting (majority vote for classification, average for regression)
    - Soft voting (average of predicted probabilities for classification)
    """
    
    def __init__(self, 
                 name: str = "Voting Ensemble",
                 description: str = "Voting ensemble combining multiple base models",
                 problem_type: ProblemType = ProblemType.BINARY_CLASSIFICATION,
                 base_estimators: Optional[List[Any]] = None,
                 voting: str = "hard",
                 **kwargs):
        """
        Initialize voting ensemble.
        
        Args:
            name: Model name
            description: Model description
            problem_type: Type of problem (classification or regression)
            base_estimators: List of base estimators
            voting: Voting strategy ("hard" or "soft")
            **kwargs: Additional configuration parameters
        """
        # Determine model type
        model_type = (ModelType.CLASSIFICATION 
                     if "classification" in problem_type.value 
                     else ModelType.REGRESSION)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=str(uuid.uuid4()),
            name=name,
            description=description,
            model_type=model_type,
            problem_type=problem_type,
            hyperparameters={
                'voting': voting,
                **kwargs
            }
        )
        
        # Create ensemble configuration
        ensemble_config = EnsembleConfig(
            ensemble_type=EnsembleType.VOTING,
            base_estimators=base_estimators or [],
            voting=voting,
            **kwargs
        )
        
        super().__init__(metadata, ensemble_config)
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'VotingEnsembleModel':
        """
        Train the voting ensemble.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Fitted ensemble model
        """
        start_time = time.time()
        self.update_status(ModelStatus.TRAINING)
        
        # Validate and prepare data
        X_validated, y_validated = self._validate_data(X, y)
        if X_validated is None or y_validated is None:
            raise ValueError("Data validation failed")
        
        # Prepare base estimators
        base_estimators = self._prepare_base_estimators()
        
        # Evaluate base models
        self._base_scores = self._evaluate_base_models(X_validated, y_validated)
        
        # Create voting ensemble
        estimator_list = [(f'estimator_{i}', est) for i, est in enumerate(base_estimators)]
        
        if self.metadata.model_type == ModelType.CLASSIFICATION:
            voting_method = "hard" if self.ensemble_config.voting == "hard" else "soft"
            self.ensemble_model = VotingClassifier(
                estimators=estimator_list,
                voting=voting_method,
                n_jobs=self.ensemble_config.n_jobs
            )
        else:
            self.ensemble_model = VotingRegressor(
                estimators=estimator_list,
                n_jobs=self.ensemble_config.n_jobs
            )
        
        # Fit the ensemble
        self.ensemble_model.fit(X_validated, y_validated)
        self.is_fitted = True
        
        # Update metadata
        training_time = time.time() - start_time
        self.metadata.training_time = training_time
        self.update_status(ModelStatus.TRAINED)
        
        # Calculate training metrics
        y_pred = self.predict(X_validated)
        if self.metadata.model_type == ModelType.CLASSIFICATION:
            accuracy = accuracy_score(y_validated, y_pred)
            self.add_metric('training_accuracy', float(accuracy))
        else:
            mse = mean_squared_error(y_validated, y_pred)
            r2 = r2_score(y_validated, y_pred)
            self.add_metric('training_mse', float(mse))
            self.add_metric('training_r2', float(r2))
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions using the voting ensemble."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X, _ = self._validate_data(X)
        predictions = self.ensemble_model.predict(X)
        return np.asarray(predictions)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Get prediction probabilities (classification only)."""
        if self.metadata.model_type != ModelType.CLASSIFICATION:
            raise ValueError("predict_proba is only available for classification")
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X, _ = self._validate_data(X)
        predict_proba_fn = getattr(self.ensemble_model, 'predict_proba', None)
        if predict_proba_fn is not None:
            return predict_proba_fn(X)
        else:
            raise ValueError("Model does not support probability predictions")


class BaggingEnsembleModel(BaseEnsembleModel):
    """
    Bagging (Bootstrap Aggregating) ensemble model.
    
    Trains multiple instances of a base estimator on different bootstrap
    samples of the training data and combines their predictions.
    """
    
    def __init__(self, 
                 name: str = "Bagging Ensemble",
                 description: str = "Bagging ensemble with bootstrap sampling",
                 problem_type: ProblemType = ProblemType.BINARY_CLASSIFICATION,
                 base_estimator: Optional[Any] = None,
                 n_estimators: int = 10,
                 max_samples: Union[int, float] = 1.0,
                 max_features: Union[int, float] = 1.0,
                 bootstrap: bool = True,
                 **kwargs):
        """
        Initialize bagging ensemble.
        
        Args:
            name: Model name
            description: Model description
            problem_type: Type of problem
            base_estimator: Base estimator to bag
            n_estimators: Number of base estimators
            max_samples: Number/fraction of samples for each base estimator
            max_features: Number/fraction of features for each base estimator
            bootstrap: Whether to use bootstrap sampling
            **kwargs: Additional parameters
        """
        model_type = (ModelType.CLASSIFICATION 
                     if "classification" in problem_type.value 
                     else ModelType.REGRESSION)
        
        metadata = ModelMetadata(
            model_id=str(uuid.uuid4()),
            name=name,
            description=description,
            model_type=model_type,
            problem_type=problem_type,
            hyperparameters={
                'n_estimators': n_estimators,
                'max_samples': max_samples,
                'max_features': max_features,
                'bootstrap': bootstrap,
                **kwargs
            }
        )
        
        ensemble_config = EnsembleConfig(
            ensemble_type=EnsembleType.BAGGING,
            base_estimators=[base_estimator] if base_estimator else [],
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            **kwargs
        )
        
        super().__init__(metadata, ensemble_config)
        
        # Set default base estimator if none provided
        if not base_estimator:
            if model_type == ModelType.CLASSIFICATION:
                self.base_estimator = DecisionTreeClassifier()
            else:
                self.base_estimator = DecisionTreeRegressor()
        else:
            self.base_estimator = base_estimator
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'BaggingEnsembleModel':
        """Train the bagging ensemble."""
        start_time = time.time()
        self.update_status(ModelStatus.TRAINING)
        
        X_validated, y_validated = self._validate_data(X, y)
        if X_validated is None or y_validated is None:
            raise ValueError("Data validation failed")
        
        # Create bagging ensemble
        if self.metadata.model_type == ModelType.CLASSIFICATION:
            self.ensemble_model = BaggingClassifier(
                base_estimator=self.base_estimator,
                n_estimators=self.ensemble_config.n_estimators,
                max_samples=self.ensemble_config.max_samples,
                max_features=self.ensemble_config.max_features,
                bootstrap=self.ensemble_config.bootstrap,
                random_state=self.ensemble_config.random_state,
                n_jobs=self.ensemble_config.n_jobs
            )
        else:
            self.ensemble_model = BaggingRegressor(
                base_estimator=self.base_estimator,
                n_estimators=self.ensemble_config.n_estimators,
                max_samples=self.ensemble_config.max_samples,
                max_features=self.ensemble_config.max_features,
                bootstrap=self.ensemble_config.bootstrap,
                random_state=self.ensemble_config.random_state,
                n_jobs=self.ensemble_config.n_jobs
            )
        
        # Fit the ensemble
        self.ensemble_model.fit(X_validated, y_validated)
        self.is_fitted = True
        
        # Update metadata
        training_time = time.time() - start_time
        self.metadata.training_time = training_time
        self.update_status(ModelStatus.TRAINED)
        
        # Calculate training metrics
        y_pred = self.predict(X_validated)
        if self.metadata.model_type == ModelType.CLASSIFICATION:
            accuracy = accuracy_score(y_validated, y_pred)
            self.add_metric('training_accuracy', float(accuracy))
        else:
            mse = mean_squared_error(y_validated, y_pred)
            r2 = r2_score(y_validated, y_pred)
            self.add_metric('training_mse', float(mse))
            self.add_metric('training_r2', float(r2))
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions using the bagging ensemble."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X, _ = self._validate_data(X)
        return self.ensemble_model.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Get prediction probabilities (classification only)."""
        if self.metadata.model_type != ModelType.CLASSIFICATION:
            raise ValueError("predict_proba is only available for classification")
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X, _ = self._validate_data(X)
        predict_proba_fn = getattr(self.ensemble_model, 'predict_proba', None)
        if predict_proba_fn is not None:
            return predict_proba_fn(X)
        else:
            raise ValueError("Model does not support probability predictions")
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from the ensemble."""
        # BaggingClassifier/Regressor don't have feature_importances_
        if hasattr(self.ensemble_model, 'feature_importances_'):
            feature_importances = getattr(self.ensemble_model, 'feature_importances_', None)
            if feature_importances is not None:
                if self.feature_names:
                    return dict(zip(self.feature_names, feature_importances))
                else:
                    return {f"feature_{i}": imp for i, imp in enumerate(feature_importances)}
        return None


class BoostingEnsembleModel(BaseEnsembleModel):
    """
    Boosting ensemble model supporting AdaBoost and Gradient Boosting.
    
    Sequentially trains weak learners, with each subsequent learner
    focusing on the mistakes of the previous ones.
    """
    
    def __init__(self, 
                 name: str = "Boosting Ensemble",
                 description: str = "Boosting ensemble with sequential learning",
                 problem_type: ProblemType = ProblemType.BINARY_CLASSIFICATION,
                 algorithm: str = "adaboost",  # "adaboost" or "gradient"
                 n_estimators: int = 50,
                 learning_rate: float = 1.0,
                 max_depth: int = 1,
                 **kwargs):
        """
        Initialize boosting ensemble.
        
        Args:
            name: Model name
            description: Model description
            problem_type: Type of problem
            algorithm: Boosting algorithm ("adaboost" or "gradient")
            n_estimators: Number of boosting stages
            learning_rate: Learning rate shrinks contribution of each classifier
            max_depth: Maximum depth of decision trees (for gradient boosting)
            **kwargs: Additional parameters
        """
        model_type = (ModelType.CLASSIFICATION 
                     if "classification" in problem_type.value 
                     else ModelType.REGRESSION)
        
        metadata = ModelMetadata(
            model_id=str(uuid.uuid4()),
            name=name,
            description=description,
            model_type=model_type,
            problem_type=problem_type,
            hyperparameters={
                'algorithm': algorithm,
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                **kwargs
            }
        )
        
        ensemble_config = EnsembleConfig(
            ensemble_type=EnsembleType.BOOSTING,
            n_estimators=n_estimators,
            **kwargs
        )
        
        super().__init__(metadata, ensemble_config)
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.max_depth = max_depth
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'BoostingEnsembleModel':
        """Train the boosting ensemble."""
        start_time = time.time()
        self.update_status(ModelStatus.TRAINING)
        
        X_validated, y_validated = self._validate_data(X, y)
        if X_validated is None or y_validated is None:
            raise ValueError("Data validation failed")
        
        # Create boosting ensemble based on algorithm
        if self.algorithm == "adaboost":
            if self.metadata.model_type == ModelType.CLASSIFICATION:
                self.ensemble_model = AdaBoostClassifier(
                    n_estimators=self.ensemble_config.n_estimators,
                    learning_rate=self.learning_rate,
                    random_state=self.ensemble_config.random_state
                )
            else:
                self.ensemble_model = AdaBoostRegressor(
                    n_estimators=self.ensemble_config.n_estimators,
                    learning_rate=self.learning_rate,
                    random_state=self.ensemble_config.random_state
                )
        
        elif self.algorithm == "gradient":
            if self.metadata.model_type == ModelType.CLASSIFICATION:
                self.ensemble_model = GradientBoostingClassifier(
                    n_estimators=self.ensemble_config.n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    random_state=self.ensemble_config.random_state
                )
            else:
                self.ensemble_model = GradientBoostingRegressor(
                    n_estimators=self.ensemble_config.n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    random_state=self.ensemble_config.random_state
                )
        
        # Fit the ensemble
        self.ensemble_model.fit(X_validated, y_validated)
        self.is_fitted = True
        
        # Update metadata
        training_time = time.time() - start_time
        self.metadata.training_time = training_time
        self.update_status(ModelStatus.TRAINED)
        
        # Calculate training metrics
        y_pred = self.predict(X_validated)
        if self.metadata.model_type == ModelType.CLASSIFICATION:
            accuracy = accuracy_score(y_validated, y_pred)
            self.add_metric('training_accuracy', float(accuracy))
        else:
            mse = mean_squared_error(y_validated, y_pred)
            r2 = r2_score(y_validated, y_pred)
            self.add_metric('training_mse', float(mse))
            self.add_metric('training_r2', float(r2))
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions using the boosting ensemble."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X, _ = self._validate_data(X)
        return self.ensemble_model.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Get prediction probabilities (classification only)."""
        if self.metadata.model_type != ModelType.CLASSIFICATION:
            raise ValueError("predict_proba is only available for classification")
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if not hasattr(self.ensemble_model, 'predict_proba'):
            raise ValueError("predict_proba not available for this boosting model")
        
        X, _ = self._validate_data(X)
        predict_proba_fn = getattr(self.ensemble_model, 'predict_proba', None)
        if predict_proba_fn is None:
            raise ValueError("predict_proba not available for this boosting model")
        return predict_proba_fn(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from the boosting ensemble."""
        if hasattr(self.ensemble_model, 'feature_importances_'):
            if self.feature_names:
                return dict(zip(self.feature_names, self.ensemble_model.feature_importances_))
            else:
                return {f"feature_{i}": imp for i, imp in enumerate(self.ensemble_model.feature_importances_)}
        return None
    
    def get_stage_scores(self) -> Optional[np.ndarray]:
        """Get staged scores during training (gradient boosting only)."""
        if hasattr(self.ensemble_model, 'train_score_'):
            return getattr(self.ensemble_model, 'train_score_', None)
        return None


class StackingEnsembleModel(BaseEnsembleModel):
    """
    Stacking ensemble model with meta-learner.
    
    Uses cross-validation to train base models and then trains a meta-learner
    on the base model predictions to make final predictions.
    """
    
    def __init__(self, 
                 name: str = "Stacking Ensemble",
                 description: str = "Stacking ensemble with meta-learner",
                 problem_type: ProblemType = ProblemType.BINARY_CLASSIFICATION,
                 base_estimators: Optional[List[Any]] = None,
                 meta_learner: Optional[Any] = None,
                 cv_folds: int = 5,
                 stack_method: str = "auto",
                 **kwargs):
        """
        Initialize stacking ensemble.
        
        Args:
            name: Model name
            description: Model description
            problem_type: Type of problem
            base_estimators: List of base estimators
            meta_learner: Meta-learner for final predictions
            cv_folds: Number of cross-validation folds
            stack_method: Method for generating meta-features
            **kwargs: Additional parameters
        """
        model_type = (ModelType.CLASSIFICATION 
                     if "classification" in problem_type.value 
                     else ModelType.REGRESSION)
        
        metadata = ModelMetadata(
            model_id=str(uuid.uuid4()),
            name=name,
            description=description,
            model_type=model_type,
            problem_type=problem_type,
            hyperparameters={
                'cv_folds': cv_folds,
                'stack_method': stack_method,
                **kwargs
            }
        )
        
        ensemble_config = EnsembleConfig(
            ensemble_type=EnsembleType.STACKING,
            base_estimators=base_estimators or [],
            cv_folds=cv_folds,
            meta_learner=meta_learner,
            stack_method=stack_method,
            **kwargs
        )
        
        super().__init__(metadata, ensemble_config)
        
        # Set default meta-learner if none provided
        if not meta_learner:
            if model_type == ModelType.CLASSIFICATION:
                self.meta_learner = LogisticRegression()
            else:
                self.meta_learner = LinearRegression()
        else:
            self.meta_learner = meta_learner
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'StackingEnsembleModel':
        """Train the stacking ensemble."""
        start_time = time.time()
        self.update_status(ModelStatus.TRAINING)
        
        X_validated, y_validated = self._validate_data(X, y)
        if X_validated is None or y_validated is None:
            raise ValueError("Data validation failed")
        
        # Prepare base estimators
        base_estimators = self._prepare_base_estimators()
        estimator_list = [(f'estimator_{i}', est) for i, est in enumerate(base_estimators)]
        
        # Import StackingClassifier/StackingRegressor
        try:
            from sklearn.ensemble import StackingClassifier, StackingRegressor
            
            if self.metadata.model_type == ModelType.CLASSIFICATION:
                # Use explicit literal matching for stack_method
                if self.ensemble_config.stack_method == "predict_proba":
                    stack_method = "predict_proba"
                elif self.ensemble_config.stack_method == "decision_function":
                    stack_method = "decision_function"
                elif self.ensemble_config.stack_method == "predict":
                    stack_method = "predict"
                else:
                    stack_method = "auto"
                
                self.ensemble_model = StackingClassifier(
                    estimators=estimator_list,
                    final_estimator=self.meta_learner,
                    cv=self.ensemble_config.cv_folds,
                    stack_method=stack_method,
                    n_jobs=self.ensemble_config.n_jobs
                )
            else:
                self.ensemble_model = StackingRegressor(
                    estimators=estimator_list,
                    final_estimator=self.meta_learner,
                    cv=self.ensemble_config.cv_folds,
                    n_jobs=self.ensemble_config.n_jobs
                )
        
        except ImportError:
            # Fallback to manual stacking implementation
            self._manual_stacking_fit(X_validated, y_validated, base_estimators)
            return self
        
        # Fit the ensemble
        self.ensemble_model.fit(X_validated, y_validated)
        self.is_fitted = True
        
        # Update metadata
        training_time = time.time() - start_time
        self.metadata.training_time = training_time
        self.update_status(ModelStatus.TRAINED)
        
        # Calculate training metrics
        y_pred = self.predict(X_validated)
        if self.metadata.model_type == ModelType.CLASSIFICATION:
            accuracy = accuracy_score(y_validated, y_pred)
            self.add_metric('training_accuracy', float(accuracy))
        else:
            mse = mean_squared_error(y_validated, y_pred)
            r2 = r2_score(y_validated, y_pred)
            self.add_metric('training_mse', float(mse))
            self.add_metric('training_r2', float(r2))
        
        return self
    
    def _manual_stacking_fit(self, X: np.ndarray, y: np.ndarray, base_estimators: List[Any]) -> None:
        """Manual implementation of stacking when sklearn version doesn't support it."""
        # Generate meta-features using cross-validation
        if self.metadata.model_type == ModelType.CLASSIFICATION:
            cv = StratifiedKFold(n_splits=self.ensemble_config.cv_folds, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=self.ensemble_config.cv_folds, shuffle=True, random_state=42)
        
        meta_features = np.zeros((X.shape[0], len(base_estimators)))
        
        for i, estimator in enumerate(base_estimators):
            cv_predictions = np.zeros(X.shape[0])
            
            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train = y[train_idx]
                
                # Clone and fit estimator
                est_clone = clone(estimator)
                est_clone.fit(X_train, y_train)
                
                # Predict on validation set
                cv_predictions[val_idx] = est_clone.predict(X_val)
            
            meta_features[:, i] = cv_predictions
        
        # Train meta-learner on meta-features
        self.meta_learner.fit(meta_features, y)
        
        # Train base estimators on full dataset
        self.base_models = []
        for estimator in base_estimators:
            est_clone = clone(estimator)
            est_clone.fit(X, y)
            self.base_models.append(est_clone)
        
        self.is_fitted = True
        self.meta_features = meta_features
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions using the stacking ensemble."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X, _ = self._validate_data(X)
        
        if hasattr(self, 'ensemble_model') and self.ensemble_model is not None:
            return self.ensemble_model.predict(X)
        else:
            # Manual prediction for manual stacking
            meta_features = np.zeros((X.shape[0], len(self.base_models)))
            
            for i, model in enumerate(self.base_models):
                meta_features[:, i] = model.predict(X)
            
            return self.meta_learner.predict(meta_features)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Get prediction probabilities (classification only)."""
        if self.metadata.model_type != ModelType.CLASSIFICATION:
            raise ValueError("predict_proba is only available for classification")
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X, _ = self._validate_data(X)
        
        if hasattr(self, 'ensemble_model') and self.ensemble_model is not None:
            predict_proba_fn = getattr(self.ensemble_model, 'predict_proba', None)
            if predict_proba_fn is None:
                raise ValueError("predict_proba not available for this stacking model")
            return np.asarray(predict_proba_fn(X))
        else:
            # Manual prediction probabilities for manual stacking
            meta_features = np.zeros((X.shape[0], len(self.base_models)))
            
            for i, model in enumerate(self.base_models):
                if hasattr(model, 'predict_proba'):
                    meta_features[:, i] = model.predict_proba(X)[:, 1]  # Take positive class probability
                else:
                    meta_features[:, i] = model.predict(X)
            
            if hasattr(self.meta_learner, 'predict_proba'):
                predict_proba_fn = getattr(self.meta_learner, 'predict_proba', None)
                if predict_proba_fn is not None:
                    return predict_proba_fn(meta_features)
            
            # Convert predictions to probabilities for regression meta-learner
            predictions = self.meta_learner.predict(meta_features)
            proba = np.column_stack([1 - predictions, predictions])
            return proba


# Factory functions for easy model creation
def create_voting_ensemble(problem_type: ProblemType = ProblemType.BINARY_CLASSIFICATION,
                          base_estimators: Optional[List[Any]] = None,
                          voting: str = "hard",
                          **kwargs) -> VotingEnsembleModel:
    """Create a voting ensemble model."""
    return VotingEnsembleModel(
        problem_type=problem_type,
        base_estimators=base_estimators,
        voting=voting,
        **kwargs
    )


def create_bagging_ensemble(problem_type: ProblemType = ProblemType.BINARY_CLASSIFICATION,
                           base_estimator: Optional[Any] = None,
                           n_estimators: int = 10,
                           **kwargs) -> BaggingEnsembleModel:
    """Create a bagging ensemble model."""
    return BaggingEnsembleModel(
        problem_type=problem_type,
        base_estimator=base_estimator,
        n_estimators=n_estimators,
        **kwargs
    )


def create_boosting_ensemble(problem_type: ProblemType = ProblemType.BINARY_CLASSIFICATION,
                            algorithm: str = "adaboost",
                            n_estimators: int = 50,
                            **kwargs) -> BoostingEnsembleModel:
    """Create a boosting ensemble model."""
    return BoostingEnsembleModel(
        problem_type=problem_type,
        algorithm=algorithm,
        n_estimators=n_estimators,
        **kwargs
    )


def create_stacking_ensemble(problem_type: ProblemType = ProblemType.BINARY_CLASSIFICATION,
                            base_estimators: Optional[List[Any]] = None,
                            meta_learner: Optional[Any] = None,
                            **kwargs) -> StackingEnsembleModel:
    """Create a stacking ensemble model."""
    return StackingEnsembleModel(
        problem_type=problem_type,
        base_estimators=base_estimators,
        meta_learner=meta_learner,
        **kwargs
    )


def create_random_forest(problem_type: ProblemType = ProblemType.BINARY_CLASSIFICATION,
                        n_estimators: int = 100,
                        max_depth: Optional[int] = None,
                        **kwargs) -> BaggingEnsembleModel:
    """Create a Random Forest ensemble (special case of bagging)."""
    model_type = ModelType.CLASSIFICATION if "classification" in problem_type.value else ModelType.REGRESSION
    
    if model_type == ModelType.CLASSIFICATION:
        base_estimator = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, **kwargs)
    else:
        base_estimator = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, **kwargs)
    
    return BaggingEnsembleModel(
        name="Random Forest Ensemble",
        description="Random Forest ensemble with decision trees",
        problem_type=problem_type,
        base_estimator=base_estimator,
        n_estimators=1  # RF handles multiple estimators internally
    )


# Export main classes and functions
__all__ = [
    # Base classes
    'BaseEnsembleModel',
    'EnsembleType',
    'EnsembleConfig',
    
    # Ensemble models
    'VotingEnsembleModel',
    'BaggingEnsembleModel', 
    'BoostingEnsembleModel',
    'StackingEnsembleModel',
    
    # Factory functions
    'create_voting_ensemble',
    'create_bagging_ensemble',
    'create_boosting_ensemble',
    'create_stacking_ensemble',
    'create_random_forest'
]
