"""
Classification Modeling Package

This module provides comprehensive classification models including logistic regression,
support vector machines, decision trees, and ensemble methods. All models inherit from
the base modeling framework and provide consistent interfaces for training, prediction,
and evaluation.
"""

import time
import uuid
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target

from ..base import (
    BaseModel, ModelMetadata, TrainingConfig, ModelType, 
    ModelStatus, ProblemType
)

logger = logging.getLogger(__name__)


class LogisticRegressionModel(BaseModel):
    def generate_report(self) -> str:
        """Generate a summary report for the logistic regression model."""
        report = [f"Model: {self.metadata.name}", f"Type: Logistic Regression", f"Description: {self.metadata.description}"]
        report.append(f"Fitted: {getattr(self, 'is_fitted', False)}")
        report.append(f"Classes: {getattr(self, 'classes_', None)}")
        report.append(f"Metrics:")
        for k, v in getattr(self, 'metrics', {}).items():
            report.append(f"  {k}: {v}")
        return '\n'.join(report)
    """
    Logistic Regression model for binary and multiclass classification.
    
    Supports L1, L2, and elastic net regularization with configurable solvers
    and penalties for different classification scenarios.
    """
    
    def __init__(
        self,
        name: str = "Logistic Regression",
        description: str = "Logistic regression classifier",
        penalty: str = 'l2',
        C: float = 1.0,
        solver: str = 'lbfgs',
        max_iter: int = 1000,
        multi_class: str = 'auto',
        class_weight: Optional[Union[str, Dict]] = None,
        random_state: Optional[int] = 42,
        **kwargs
    ):
        """
        Initialize logistic regression model.
        
        Args:
            name: Model name
            description: Model description
            penalty: Regularization penalty ('l1', 'l2', 'elasticnet', 'none')
            C: Inverse regularization strength
            solver: Algorithm for optimization
            max_iter: Maximum number of iterations
            multi_class: Multi-class strategy ('auto', 'ovr', 'multinomial')
            class_weight: Weights for classes
            random_state: Random seed
        """
        # Determine problem type based on multi_class setting
        problem_type = (ProblemType.BINARY_CLASSIFICATION if multi_class == 'ovr' 
                       else ProblemType.MULTICLASS_CLASSIFICATION)
        
        metadata = ModelMetadata(
            model_id=str(uuid.uuid4()),
            name=name,
            description=description,
            model_type=ModelType.CLASSIFICATION,
            problem_type=problem_type,
            hyperparameters={
                'penalty': penalty,
                'C': C,
                'solver': solver,
                'max_iter': max_iter,
                'multi_class': multi_class,
                'class_weight': class_weight,
                'random_state': random_state,
                **kwargs
            }
        )
        
        super().__init__(metadata)
        
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.class_weight = class_weight
        self.random_state = random_state
        self.classes_ = None
        self.n_classes_ = None
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'LogisticRegressionModel':
        """
        Train the logistic regression model.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Fitted model instance
        """
        start_time = time.time()
        
        # Validate and prepare data
        X_processed, y_processed = self._validate_data(X, y)
        
        # Ensure y_processed is not None
        if y_processed is None:
            raise ValueError("Target values (y) must not be None.")
        
        # Determine problem type from target
        target_type = type_of_target(y_processed)
        if target_type == 'binary':
            self.metadata.problem_type = ProblemType.BINARY_CLASSIFICATION
        elif target_type in ['multiclass', 'multiclass-multioutput']:
            self.metadata.problem_type = ProblemType.MULTICLASS_CLASSIFICATION
        
        # Store class information
        self.classes_ = np.unique(y_processed)
        self.n_classes_ = len(self.classes_)
        
        # Create and train sklearn model
        self._model = LogisticRegression(
            penalty=self.penalty if self.penalty in ('l1', 'l2', 'elasticnet', None) else 'l2',
            C=self.C,
            solver=self.solver if self.solver in ('lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga') else 'lbfgs',
            max_iter=self.max_iter,
            multi_class=self.multi_class if self.multi_class in ('auto', 'ovr', 'multinomial') else 'auto',
            class_weight=self.class_weight,
            random_state=self.random_state
        )
        
        self._model.fit(X_processed, y_processed)
        
        # Update metadata
        training_time = time.time() - start_time
        self.metadata.training_time = training_time
        self.update_status(ModelStatus.TRAINED)
        self.is_fitted = True
        
        # Calculate training metrics
        y_pred = self._model.predict(X_processed)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = float(accuracy_score(y_processed, y_pred))
        precision = float(precision_score(y_processed, y_pred, average='weighted'))
        recall = float(recall_score(y_processed, y_pred, average='weighted'))
        f1 = float(f1_score(y_processed, y_pred, average='weighted'))
        
        self.add_metric('accuracy', accuracy)
        self.add_metric('precision', precision)
        self.add_metric('recall', recall)
        self.add_metric('f1_score', f1)
        
        # Add class-specific metrics for binary classification
        if self.n_classes_ == 2:
            precision_binary = float(precision_score(y_processed, y_pred, pos_label=self.classes_[1]))
            recall_binary = float(recall_score(y_processed, y_pred, pos_label=self.classes_[1]))
            f1_binary = float(f1_score(y_processed, y_pred, pos_label=self.classes_[1]))
            
            self.add_metric('precision_binary', precision_binary)
            self.add_metric('recall_binary', recall_binary)
            self.add_metric('f1_binary', f1_binary)
        
        logger.info(f"Logistic regression trained in {training_time:.4f}s with accuracy: {accuracy:.4f}")
        
        return self
        
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_processed, _ = self._validate_data(X)
        return self._model.predict(X_processed)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predicted class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_processed, _ = self._validate_data(X)
        proba = self._model.predict_proba(X_processed)
        if isinstance(proba, list):
            # Some models (e.g., multioutput) may return a list of arrays
            proba = np.array(proba)
        return proba
    
    def get_coefficients(self) -> Dict[str, Union[float, np.ndarray]]:
        """Get model coefficients."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get coefficients")
        
        coefficients = {}
        
        if self.feature_names:
            if self.n_classes_ == 2:
                # Binary classification - single coefficient vector
                coefficients.update(dict(zip(self.feature_names, self._model.coef_[0])))
            else:
                # Multiclass - coefficient matrix
                if self.classes_ is not None:
                    for i, class_name in enumerate(self.classes_):
                        for j, feature_name in enumerate(self.feature_names):
                            coefficients[f'{feature_name}_class_{class_name}'] = self._model.coef_[i, j]
        
        # Add intercept
        if self.n_classes_ == 2:
            coefficients['intercept'] = self._model.intercept_[0]
        else:
            if self.classes_ is not None:
                for i, class_name in enumerate(self.classes_):
                    coefficients[f'intercept_class_{class_name}'] = self._model.intercept_[i]
        
        return coefficients


class SVMModel(BaseModel):
    def generate_report(self) -> str:
        """Generate a summary report for the SVM model."""
        report = [f"Model: {self.metadata.name}", f"Type: SVM", f"Description: {self.metadata.description}"]
        report.append(f"Fitted: {getattr(self, 'is_fitted', False)}")
        report.append(f"Classes: {getattr(self, 'classes_', None)}")
        report.append(f"Metrics:")
        for k, v in getattr(self, 'metrics', {}).items():
            report.append(f"  {k}: {v}")
        return '\n'.join(report)
    """
    Support Vector Machine model for classification.
    
    Supports different kernels (linear, polynomial, RBF, sigmoid) and 
    various SVM parameters for both binary and multiclass classification.
    """
    
    def __init__(
        self,
        name: str = "SVM Classifier",
        description: str = "Support Vector Machine classifier",
        kernel: str = 'rbf',
        C: float = 1.0,
        gamma: Union[str, float] = 'scale',
        degree: int = 3,
        coef0: float = 0.0,
        class_weight: Optional[Union[str, Dict]] = None,
        probability: bool = True,
        random_state: Optional[int] = 42,
        **kwargs
    ):
        """
        Initialize SVM model.
        
        Args:
            name: Model name
            description: Model description
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            C: Regularization parameter
            gamma: Kernel coefficient
            degree: Degree for polynomial kernel
            coef0: Independent term in kernel function
            class_weight: Weights for classes
            probability: Enable probability estimates
            random_state: Random seed
        """
        metadata = ModelMetadata(
            model_id=str(uuid.uuid4()),
            name=name,
            description=description,
            model_type=ModelType.CLASSIFICATION,
            problem_type=ProblemType.BINARY_CLASSIFICATION,
            hyperparameters={
                'kernel': kernel,
                'C': C,
                'gamma': gamma,
                'degree': degree,
                'coef0': coef0,
                'class_weight': class_weight,
                'probability': probability,
                'random_state': random_state,
                **kwargs
            }
        )
        
        super().__init__(metadata)
        
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.class_weight = class_weight
        self.probability = probability
        self.random_state = random_state
        self.classes_ = None
        self.n_classes_ = None
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'SVMModel':
        """
        Train the SVM model.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Fitted model instance
        """
        start_time = time.time()
        
        # Validate and prepare data
        X_processed, y_processed = self._validate_data(X, y)

        # Ensure y_processed is not None
        if y_processed is None:
            raise ValueError("Target values (y) must not be None.")

        # Determine problem type
        target_type = type_of_target(y_processed)
        if target_type == 'binary':
            self.metadata.problem_type = ProblemType.BINARY_CLASSIFICATION
        elif target_type in ['multiclass', 'multiclass-multioutput']:
            self.metadata.problem_type = ProblemType.MULTICLASS_CLASSIFICATION

        # Store class information
        self.classes_ = np.unique(y_processed)
        self.n_classes_ = len(self.classes_)

        # Validate kernel and gamma types for SVC
        valid_kernels = ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed')
        kernel = self.kernel if self.kernel in valid_kernels else 'rbf'
        valid_gamma = ('scale', 'auto')
        gamma = self.gamma if (isinstance(self.gamma, float) or self.gamma in valid_gamma) else 'scale'

        # Create and train sklearn model
        self._model = SVC(
            kernel=kernel,
            C=self.C,
            gamma=gamma,
            degree=self.degree,
            coef0=self.coef0,
            class_weight=self.class_weight,
            probability=self.probability,
            random_state=self.random_state
        )

        self._model.fit(X_processed, y_processed)

        # Update metadata
        training_time = time.time() - start_time
        self.metadata.training_time = training_time
        self.update_status(ModelStatus.TRAINED)
        self.is_fitted = True

        # Calculate training metrics
        y_pred = self._model.predict(X_processed)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        accuracy = float(accuracy_score(y_processed, y_pred))
        precision = float(precision_score(y_processed, y_pred, average='weighted'))
        recall = float(recall_score(y_processed, y_pred, average='weighted'))
        f1 = float(f1_score(y_processed, y_pred, average='weighted'))

        self.add_metric('accuracy', accuracy)
        self.add_metric('precision', precision)
        self.add_metric('recall', recall)
        self.add_metric('f1_score', f1)
        self.add_metric('n_support_vectors', int(np.sum(self._model.n_support_)))

        logger.info(f"SVM trained in {training_time:.4f}s with accuracy: {accuracy:.4f}")

        return self
        
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_processed, _ = self._validate_data(X)
        return self._model.predict(X_processed)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if not self.probability:
            raise ValueError("Probability estimation not enabled. Set probability=True when creating the model.")
        
        X_processed, _ = self._validate_data(X)
        proba = self._model.predict_proba(X_processed)
        if isinstance(proba, list):
            proba = np.array(proba)
        return proba
    
    def get_support_vectors(self) -> Optional[np.ndarray]:
        """Get support vectors."""
        if not self.is_fitted:
            return None
        return self._model.support_vectors_


class DecisionTreeModel(BaseModel):
    def generate_report(self) -> str:
        """Generate a summary report for the decision tree model."""
        report = [f"Model: {self.metadata.name}", f"Type: Decision Tree", f"Description: {self.metadata.description}"]
        report.append(f"Fitted: {getattr(self, 'is_fitted', False)}")
        report.append(f"Classes: {getattr(self, 'classes_', None)}")
        report.append(f"Metrics:")
        for k, v in getattr(self, 'metrics', {}).items():
            report.append(f"  {k}: {v}")
        return '\n'.join(report)
    """
    Decision Tree model for classification.
    
    Provides interpretable tree-based classification with configurable
    splitting criteria, pruning parameters, and tree structure controls.
    """
    
    def __init__(
        self,
        name: str = "Decision Tree",
        description: str = "Decision tree classifier",
        criterion: str = 'gini',
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        max_features: Optional[Union[int, float, str]] = None,
        class_weight: Optional[Union[str, Dict]] = None,
        random_state: Optional[int] = 42,
        **kwargs
    ):
        """
        Initialize decision tree model.
        
        Args:
            name: Model name
            description: Model description
            criterion: Splitting criterion ('gini', 'entropy', 'log_loss')
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split node
            min_samples_leaf: Minimum samples in leaf
            max_features: Number of features for best split
            class_weight: Weights for classes
            random_state: Random seed
        """
        metadata = ModelMetadata(
            model_id=str(uuid.uuid4()),
            name=name,
            description=description,
            model_type=ModelType.CLASSIFICATION,
            problem_type=ProblemType.BINARY_CLASSIFICATION,
            hyperparameters={
                'criterion': criterion,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'max_features': max_features,
                'class_weight': class_weight,
                'random_state': random_state,
                **kwargs
            }
        )
        
        super().__init__(metadata)
        
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state
        self.classes_ = None
        self.n_classes_ = None
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'DecisionTreeModel':
        """Train the decision tree model."""
        start_time = time.time()
        
        # Validate and prepare data
        X_processed, y_processed = self._validate_data(X, y)

        # Ensure y_processed is not None
        if y_processed is None:
            raise ValueError("Target values (y) must not be None.")

        # Determine problem type
        target_type = type_of_target(y_processed)
        if target_type == 'binary':
            self.metadata.problem_type = ProblemType.BINARY_CLASSIFICATION
        elif target_type in ['multiclass', 'multiclass-multioutput']:
            self.metadata.problem_type = ProblemType.MULTICLASS_CLASSIFICATION

        # Store class information
        self.classes_ = np.unique(y_processed)
        self.n_classes_ = len(self.classes_)

        # Validate criterion
        valid_criteria = ('gini', 'entropy', 'log_loss')
        criterion = self.criterion if self.criterion in valid_criteria else 'gini'

        # Validate max_features
        valid_max_features = ('auto', 'sqrt', 'log2')
        max_features = self.max_features
        if isinstance(max_features, str) and max_features not in valid_max_features:
            max_features = None

        self._model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=max_features,
            class_weight=self.class_weight,
            random_state=self.random_state
        )

        self._model.fit(X_processed, y_processed)

        # Update metadata
        training_time = time.time() - start_time
        self.metadata.training_time = training_time
        self.update_status(ModelStatus.TRAINED)
        self.is_fitted = True

        # Calculate training metrics
        y_pred = self._model.predict(X_processed)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        accuracy = float(accuracy_score(y_processed, y_pred))
        precision = float(precision_score(y_processed, y_pred, average='weighted'))
        recall = float(recall_score(y_processed, y_pred, average='weighted'))
        f1 = float(f1_score(y_processed, y_pred, average='weighted'))

        self.add_metric('accuracy', accuracy)
        self.add_metric('precision', precision)
        self.add_metric('recall', recall)
        self.add_metric('f1_score', f1)
        self.add_metric('tree_depth', int(self._model.get_depth()))
        self.add_metric('n_nodes', int(self._model.tree_.node_count))
        self.add_metric('n_leaves', int(self._model.get_n_leaves()))

        logger.info(f"Decision tree trained in {training_time:.4f}s with accuracy: {accuracy:.4f}")

        return self
        
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_processed, _ = self._validate_data(X)
        return self._model.predict(X_processed)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_processed, _ = self._validate_data(X)
        proba = self._model.predict_proba(X_processed)
        if isinstance(proba, list):
            proba = np.array(proba)
        return proba
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from the trained tree."""
        if not self.is_fitted:
            return None
        
        if self.feature_names:
            return dict(zip(self.feature_names, self._model.feature_importances_))
        else:
            return {f"feature_{i}": imp for i, imp in enumerate(self._model.feature_importances_)}


class RandomForestModel(BaseModel):
    def generate_report(self) -> str:
        """Generate a summary report for the random forest model."""
        report = [f"Model: {self.metadata.name}", f"Type: Random Forest", f"Description: {self.metadata.description}"]
        report.append(f"Fitted: {getattr(self, 'is_fitted', False)}")
        report.append(f"Classes: {getattr(self, 'classes_', None)}")
        report.append(f"Metrics:")
        for k, v in getattr(self, 'metrics', {}).items():
            report.append(f"  {k}: {v}")
        return '\n'.join(report)
    """
    Random Forest model for classification.
    
    Ensemble method combining multiple decision trees with bootstrap
    aggregating and random feature selection for improved accuracy
    and reduced overfitting.
    """
    
    def __init__(
        self,
        name: str = "Random Forest",
        description: str = "Random forest classifier",
        n_estimators: int = 100,
        criterion: str = 'gini',
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        max_features: Union[int, float, str] = 'sqrt',
        bootstrap: bool = True,
        class_weight: Optional[Union[str, Dict]] = None,
        random_state: Optional[int] = 42,
        n_jobs: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize random forest model.
        
        Args:
            name: Model name
            description: Model description
            n_estimators: Number of trees
            criterion: Splitting criterion
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split node
            min_samples_leaf: Minimum samples in leaf
            max_features: Number of features for best split
            bootstrap: Whether to bootstrap samples
            class_weight: Weights for classes
            random_state: Random seed
            n_jobs: Number of parallel jobs
        """
        metadata = ModelMetadata(
            model_id=str(uuid.uuid4()),
            name=name,
            description=description,
            model_type=ModelType.CLASSIFICATION,
            problem_type=ProblemType.BINARY_CLASSIFICATION,
            hyperparameters={
                'n_estimators': n_estimators,
                'criterion': criterion,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'max_features': max_features,
                'bootstrap': bootstrap,
                'class_weight': class_weight,
                'random_state': random_state,
                'n_jobs': n_jobs,
                **kwargs
            }
        )
        
        super().__init__(metadata)
        
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.classes_ = None
        self.n_classes_ = None
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'RandomForestModel':
        """Train the random forest model."""
        start_time = time.time()
        
        # Validate and prepare data
        X_processed, y_processed = self._validate_data(X, y)

        # Ensure y_processed is not None
        if y_processed is None:
            raise ValueError("Target values (y) must not be None.")

        # Determine problem type
        target_type = type_of_target(y_processed)
        if target_type == 'binary':
            self.metadata.problem_type = ProblemType.BINARY_CLASSIFICATION
        elif target_type in ['multiclass', 'multiclass-multioutput']:
            self.metadata.problem_type = ProblemType.MULTICLASS_CLASSIFICATION

        # Store class information
        self.classes_ = np.unique(y_processed)
        self.n_classes_ = len(self.classes_)

        # Validate criterion
        valid_criteria = ('gini', 'entropy', 'log_loss')
        criterion = self.criterion if self.criterion in valid_criteria else 'gini'

        # Validate max_features
        valid_max_features = ('sqrt', 'log2')
        max_features = self.max_features
        if isinstance(max_features, str) and max_features not in valid_max_features:
            max_features = 'sqrt'
        elif max_features is None:
            max_features = 'sqrt'

        # Validate class_weight
        valid_class_weights = ('balanced', 'balanced_subsample')
        class_weight = self.class_weight
        if isinstance(class_weight, str) and class_weight not in valid_class_weights:
            class_weight = None

        # Create and train sklearn model
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=max_features,
            bootstrap=self.bootstrap,
            class_weight=class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )

        self._model.fit(X_processed, y_processed)

        # Update metadata
        training_time = time.time() - start_time
        self.metadata.training_time = training_time
        self.update_status(ModelStatus.TRAINED)
        self.is_fitted = True

        # Calculate training metrics
        y_pred = self._model.predict(X_processed)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        accuracy = float(accuracy_score(y_processed, y_pred))
        precision = float(precision_score(y_processed, y_pred, average='weighted'))
        recall = float(recall_score(y_processed, y_pred, average='weighted'))
        f1 = float(f1_score(y_processed, y_pred, average='weighted'))

        self.add_metric('accuracy', accuracy)
        self.add_metric('precision', precision)
        self.add_metric('recall', recall)
        self.add_metric('f1_score', f1)
        self.add_metric('oob_score', float(self._model.oob_score_) if hasattr(self._model, 'oob_score_') else 0.0)

        logger.info(f"Random forest trained in {training_time:.4f}s with accuracy: {accuracy:.4f}")

        return self
        
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_processed, _ = self._validate_data(X)
        return self._model.predict(X_processed)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_processed, _ = self._validate_data(X)
        return self._model.predict_proba(X_processed)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from the ensemble."""
        if not self.is_fitted:
            return None
        
        if self.feature_names:
            return dict(zip(self.feature_names, self._model.feature_importances_))
        else:
            return {f"feature_{i}": imp for i, imp in enumerate(self._model.feature_importances_)}


class ClassificationModelFactory:
    """Factory class for creating classification models."""
    
    @staticmethod
    def create_logistic_regression(
        penalty: str = 'l2',
        C: float = 1.0,
        solver: str = 'lbfgs',
        **kwargs
    ) -> LogisticRegressionModel:
        """Create a logistic regression model."""
        # Extract name from kwargs to avoid conflicts
        name = kwargs.pop('name', "Logistic Regression")
        return LogisticRegressionModel(
            name=name,
            penalty=penalty,
            C=C,
            solver=solver,
            **kwargs
        )
    
    @staticmethod
    def create_svm(
        kernel: str = 'rbf',
        C: float = 1.0,
        gamma: Union[str, float] = 'scale',
        **kwargs
    ) -> SVMModel:
        """Create an SVM model."""
        name = kwargs.pop('name', "SVM Classifier")
        return SVMModel(
            name=name,
            kernel=kernel,
            C=C,
            gamma=gamma,
            **kwargs
        )
    
    @staticmethod
    def create_decision_tree(
        criterion: str = 'gini',
        max_depth: Optional[int] = None,
        **kwargs
    ) -> DecisionTreeModel:
        """Create a decision tree model."""
        name = kwargs.pop('name', "Decision Tree")
        return DecisionTreeModel(
            name=name,
            criterion=criterion,
            max_depth=max_depth,
            **kwargs
        )
    
    @staticmethod
    def create_random_forest(
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        **kwargs
    ) -> RandomForestModel:
        """Create a random forest model."""
        name = kwargs.pop('name', "Random Forest")
        return RandomForestModel(
            name=name,
            n_estimators=n_estimators,
            max_depth=max_depth,
            **kwargs
        )


# Convenience functions for easy model creation
def create_logistic_regression(**kwargs) -> LogisticRegressionModel:
    """Create a logistic regression model with default parameters."""
    return ClassificationModelFactory.create_logistic_regression(**kwargs)

def create_svm(**kwargs) -> SVMModel:
    """Create an SVM model with default parameters."""
    return ClassificationModelFactory.create_svm(**kwargs)

def create_decision_tree(**kwargs) -> DecisionTreeModel:
    """Create a decision tree model with default parameters."""
    return ClassificationModelFactory.create_decision_tree(**kwargs)

def create_random_forest(**kwargs) -> RandomForestModel:
    """Create a random forest model with default parameters."""
    return ClassificationModelFactory.create_random_forest(**kwargs)


# Export all public classes and functions
__all__ = [
    'LogisticRegressionModel',
    'SVMModel', 
    'DecisionTreeModel',
    'RandomForestModel',
    'ClassificationModelFactory',
    'create_logistic_regression',
    'create_svm',
    'create_decision_tree', 
    'create_random_forest'
]
