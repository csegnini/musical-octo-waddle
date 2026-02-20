"""
Regression Models Package

This package provides comprehensive regression modeling capabilities including
linear regression, polynomial regression, regularized regression, and ensemble methods.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

from ..base import BaseModel, ModelMetadata, TrainingConfig, ModelType, ModelStatus, ProblemType

logger = logging.getLogger(__name__)


class LinearRegressionModel(BaseModel):
    """
    Linear regression model implementation.
    
    This class provides a comprehensive linear regression implementation with
    support for regularization, cross-validation, and detailed performance metrics.
    """
    
    def __init__(
        self,
        name: str = "Linear Regression",
        description: str = "Linear regression model for continuous target prediction",
        regularization: Optional[str] = None,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        normalize: bool = False,
        **kwargs
    ):
        """
        Initialize linear regression model.
        
        Args:
            name: Model name
            description: Model description
            regularization: Type of regularization ('ridge', 'lasso', 'elastic_net')
            alpha: Regularization strength
            fit_intercept: Whether to fit intercept
            normalize: Whether to normalize features
            **kwargs: Additional hyperparameters
        """
        
        # Create metadata
        problem_type = ProblemType.LINEAR_REGRESSION
        metadata = ModelMetadata(
            model_id=f"linear_reg_{hash(name)}",
            name=name,
            description=description,
            model_type=ModelType.REGRESSION,
            problem_type=problem_type,
            hyperparameters={
                'regularization': regularization,
                'alpha': alpha,
                'fit_intercept': fit_intercept,
                'normalize': normalize,
                **kwargs
            }
        )
        
        super().__init__(metadata)
        
        # Initialize the underlying sklearn model
        self.regularization = regularization
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the sklearn model based on regularization type."""
        if self.regularization == 'ridge':
            self._model = Ridge(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept
            )
        elif self.regularization == 'lasso':
            self._model = Lasso(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept
            )
        elif self.regularization == 'elastic_net':
            l1_ratio = self.metadata.hyperparameters.get('l1_ratio', 0.5)
            self._model = ElasticNet(
                alpha=self.alpha,
                l1_ratio=l1_ratio,
                fit_intercept=self.fit_intercept
            )
        else:
            self._model = LinearRegression(
                fit_intercept=self.fit_intercept
            )
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'LinearRegressionModel':
        """
        Train the linear regression model.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Fitted model instance
        """
        try:
            # Validate and prepare data
            X_array, y_array = self._validate_data(X, y)
            
            if y_array is None:
                raise ValueError("Target values (y) are required for training")
            
            # Store data info
            self.data_info = self._get_data_info(X_array, y_array)
            self.metadata.features = self.feature_names or []
            self.metadata.target = self.target_name
            
            # Update status
            self.update_status(ModelStatus.TRAINING)
            
            # Fit the model
            start_time = pd.Timestamp.now()
            self._model.fit(X_array, y_array)
            end_time = pd.Timestamp.now()
            
            # Store training time
            self.metadata.training_time = (end_time - start_time).total_seconds()
            
            # Mark as fitted
            self.is_fitted = True
            self.update_status(ModelStatus.TRAINED)
            
            # Calculate and store metrics
            y_pred = self._model.predict(X_array)
            self._calculate_training_metrics(y_array, y_pred)
            
            logger.info(f"Linear regression model '{self.metadata.name}' fitted successfully")
            return self
            
        except Exception as e:
            logger.error(f"Error fitting linear regression model: {e}")
            self.update_status(ModelStatus.UNTRAINED)
            raise
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Validate data (no target)
            X_array, _ = self._validate_data(X)
            
            # Make predictions
            predictions = self._model.predict(X_array)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def predict_with_confidence(self, X: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals (if supported).
        
        Args:
            X: Features for prediction
            
        Returns:
            Tuple of (predictions, confidence_intervals)
        """
        predictions = self.predict(X)
        
        # For linear regression, we can estimate confidence intervals
        # using the prediction variance (simplified approach)
        if hasattr(self._model, 'predict') and hasattr(self._model, 'coef_'):
            # Simple confidence interval estimation
            X_array, _ = self._validate_data(X)
            
            # Calculate prediction variance (simplified)
            mse = self.metadata.metrics.get('mse', 1.0)
            confidence_intervals = np.full_like(predictions, np.sqrt(mse) * 1.96)  # 95% CI
            
            return predictions, confidence_intervals
        
        return predictions, np.zeros_like(predictions)
    
    def get_coefficients(self) -> Dict[str, float]:
        """Get model coefficients."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get coefficients")
        
        coefficients = {}
        
        if hasattr(self._model, 'coef_'):
            feature_names = self.feature_names or [f"feature_{i}" for i in range(len(self._model.coef_))]
            coefficients = dict(zip(feature_names, self._model.coef_))
        
        if hasattr(self._model, 'intercept_'):
            coefficients['intercept'] = self._model.intercept_
        
        return coefficients
    
    def _calculate_training_metrics(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculate and store training metrics."""
        try:
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mse)
            
            self.add_metric('mse', mse)
            self.add_metric('mae', mae)
            self.add_metric('r2', r2)
            self.add_metric('rmse', rmse)
            
            # Calculate adjusted R²
            if hasattr(self, 'data_info'):
                n = self.data_info.n_samples
                p = self.data_info.n_features
                adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
                self.add_metric('adj_r2', adj_r2)
            
        except Exception as e:
            logger.warning(f"Error calculating training metrics: {e}")


class PolynomialRegressionModel(BaseModel):
    """
    Polynomial regression model implementation.
    
    This class provides polynomial regression by combining polynomial feature
    transformation with linear regression.
    """
    
    def __init__(
        self,
        degree: int = 2,
        name: str = "Polynomial Regression",
        description: str = "Polynomial regression model for non-linear relationships",
        include_bias: bool = True,
        interaction_only: bool = False,
        regularization: Optional[str] = None,
        alpha: float = 1.0,
        **kwargs
    ):
        """
        Initialize polynomial regression model.
        
        Args:
            degree: Degree of polynomial features
            name: Model name
            description: Model description
            include_bias: Whether to include bias column
            interaction_only: Whether to include only interaction features
            regularization: Type of regularization
            alpha: Regularization strength
            **kwargs: Additional hyperparameters
        """
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=f"poly_reg_{hash(name)}",
            name=name,
            description=description,
            model_type=ModelType.REGRESSION,
            problem_type=ProblemType.POLYNOMIAL_REGRESSION,
            hyperparameters={
                'degree': degree,
                'include_bias': include_bias,
                'interaction_only': interaction_only,
                'regularization': regularization,
                'alpha': alpha,
                **kwargs
            }
        )
        
        super().__init__(metadata)
        
        # Initialize components
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.regularization = regularization
        self.alpha = alpha
        
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the polynomial regression pipeline."""
        # Polynomial features transformer
        poly_features = PolynomialFeatures(
            degree=self.degree,
            include_bias=self.include_bias,
            interaction_only=self.interaction_only
        )
        
        # Linear regression model
        if self.regularization == 'ridge':
            regressor = Ridge(alpha=self.alpha)
        elif self.regularization == 'lasso':
            regressor = Lasso(alpha=self.alpha)
        elif self.regularization == 'elastic_net':
            l1_ratio = self.metadata.hyperparameters.get('l1_ratio', 0.5)
            regressor = ElasticNet(alpha=self.alpha, l1_ratio=l1_ratio)
        else:
            regressor = LinearRegression()
        
        # Create pipeline
        self._model = Pipeline([
            ('poly_features', poly_features),
            ('regressor', regressor)
        ])
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'PolynomialRegressionModel':
        """
        Train the polynomial regression model.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Fitted model instance
        """
        try:
            # Validate and prepare data
            X_array, y_array = self._validate_data(X, y)
            
            if y_array is None:
                raise ValueError("Target values (y) are required for training")
            
            # Store data info
            self.data_info = self._get_data_info(X_array, y_array)
            self.metadata.features = self.feature_names or []
            self.metadata.target = self.target_name
            
            # Update status
            self.update_status(ModelStatus.TRAINING)
            
            # Fit the pipeline
            start_time = pd.Timestamp.now()
            self._model.fit(X_array, y_array)
            end_time = pd.Timestamp.now()
            
            # Store training time
            self.metadata.training_time = (end_time - start_time).total_seconds()
            
            # Mark as fitted
            self.is_fitted = True
            self.update_status(ModelStatus.TRAINED)
            
            # Calculate and store metrics
            y_pred = self._model.predict(X_array)
            assert isinstance(y_pred, np.ndarray), "Predictions should be numpy array"
            self._calculate_training_metrics(y_array, y_pred)
            
            logger.info(f"Polynomial regression model '{self.metadata.name}' fitted successfully")
            return self
            
        except Exception as e:
            logger.error(f"Error fitting polynomial regression model: {e}")
            self.update_status(ModelStatus.UNTRAINED)
            raise
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Validate data
            X_array, _ = self._validate_data(X)
            
            # Make predictions
            predictions = self._model.predict(X_array)
            assert isinstance(predictions, np.ndarray), "Predictions should be numpy array"
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def get_feature_names_out(self) -> List[str]:
        """Get the names of the polynomial features."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature names")
        
        poly_transformer = self._model.named_steps['poly_features']
        if hasattr(poly_transformer, 'get_feature_names_out'):
            feature_names = self.feature_names or [f"x{i}" for i in range(len(self.feature_names or []))]
            return poly_transformer.get_feature_names_out(feature_names).tolist()
        
        return []
    
    def _calculate_training_metrics(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculate and store training metrics."""
        try:
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mse)
            
            self.add_metric('mse', mse)
            self.add_metric('mae', mae)
            self.add_metric('r2', r2)
            self.add_metric('rmse', rmse)
            
            # Calculate adjusted R²
            if hasattr(self, 'data_info'):
                n = self.data_info.n_samples
                # For polynomial regression, p is the number of polynomial features
                poly_transformer = self._model.named_steps['poly_features']
                if hasattr(poly_transformer, 'n_output_features_'):
                    p = poly_transformer.n_output_features_
                else:
                    p = self.data_info.n_features  # Fallback
                
                adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
                self.add_metric('adj_r2', adj_r2)
            
        except Exception as e:
            logger.warning(f"Error calculating training metrics: {e}")


class RegressionModelFactory:
    """
    Factory class for creating regression models.
    """
    
    @staticmethod
    def create_linear_regression(
        regularization: Optional[str] = None,
        alpha: float = 1.0,
        **kwargs
    ) -> LinearRegressionModel:
        """Create a linear regression model."""
        return LinearRegressionModel(
            regularization=regularization,
            alpha=alpha,
            **kwargs
        )
    
    @staticmethod
    def create_polynomial_regression(
        degree: int = 2,
        regularization: Optional[str] = None,
        alpha: float = 1.0,
        **kwargs
    ) -> PolynomialRegressionModel:
        """Create a polynomial regression model."""
        return PolynomialRegressionModel(
            degree=degree,
            regularization=regularization,
            alpha=alpha,
            **kwargs
        )
    
    @staticmethod
    def create_ridge_regression(alpha: float = 1.0, **kwargs) -> LinearRegressionModel:
        """Create a ridge regression model."""
        # Extract name from kwargs to avoid conflicts
        name = kwargs.pop('name', "Ridge Regression")
        return LinearRegressionModel(
            name=name,
            description="Ridge regression with L2 regularization",
            regularization="ridge",
            alpha=alpha,
            **kwargs
        )
    
    @staticmethod
    def create_lasso_regression(alpha: float = 1.0, **kwargs) -> LinearRegressionModel:
        """Create a lasso regression model."""
        # Extract name from kwargs to avoid conflicts
        name = kwargs.pop('name', "Lasso Regression")
        return LinearRegressionModel(
            name=name,
            description="Lasso regression with L1 regularization",
            regularization="lasso",
            alpha=alpha,
            **kwargs
        )
    
    @staticmethod
    def create_elastic_net(alpha: float = 1.0, l1_ratio: float = 0.5, **kwargs) -> LinearRegressionModel:
        """Create an elastic net regression model."""
        # Extract name from kwargs to avoid conflicts
        name = kwargs.pop('name', "Elastic Net Regression")
        return LinearRegressionModel(
            name=name,
            description="Elastic Net regression with L1 and L2 regularization",
            regularization="elastic_net",
            alpha=alpha,
            l1_ratio=l1_ratio,
            **kwargs
        )


# Convenience functions for easy model creation
def create_linear_regression(**kwargs) -> LinearRegressionModel:
    """Create a linear regression model with default parameters."""
    return RegressionModelFactory.create_linear_regression(**kwargs)

def create_polynomial_regression(**kwargs) -> PolynomialRegressionModel:
    """Create a polynomial regression model with default parameters."""
    return RegressionModelFactory.create_polynomial_regression(**kwargs)

def create_ridge_regression(**kwargs) -> LinearRegressionModel:
    """Create a ridge regression model with default parameters."""
    return RegressionModelFactory.create_ridge_regression(**kwargs)

def create_lasso_regression(**kwargs) -> LinearRegressionModel:
    """Create a lasso regression model with default parameters."""
    return RegressionModelFactory.create_lasso_regression(**kwargs)

def create_elastic_net(**kwargs) -> LinearRegressionModel:
    """Create an elastic net regression model with default parameters."""
    return RegressionModelFactory.create_elastic_net(**kwargs)


# Export all public classes and functions
__all__ = [
    'LinearRegressionModel',
    'PolynomialRegressionModel', 
    'RegressionModelFactory',
    'create_linear_regression',
    'create_polynomial_regression',
    'create_ridge_regression',
    'create_lasso_regression',
    'create_elastic_net'
]
