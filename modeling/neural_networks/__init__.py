"""
Neural Networks Package

This module provides comprehensive deep learning capabilities including:
- Feedforward Neural Networks (Multi-layer Perceptrons)
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs/LSTMs)
- Autoencoders for dimensionality reduction
- Custom layers and activation functions
- Advanced optimizers and training utilities
- Model visualization and analysis tools

All neural network models integrate seamlessly with the base modeling framework
and support both classification and regression tasks.
"""

import uuid
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
import warnings

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    import keras
    from keras import layers, models, optimizers, callbacks
    from keras.utils import plot_model
    TENSORFLOW_AVAILABLE = True
    print("✅ TensorFlow/Keras available for neural networks")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    # Create dummy modules for type checking
    tf = None
    keras = None
    layers = None
    models = None
    optimizers = None
    callbacks = None
    plot_model = None
    print("❌ TensorFlow not available - using sklearn MLPClassifier/Regressor as fallback")

# Always import sklearn as fallback
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

# Import base classes
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from base import BaseModel, ModelMetadata, ModelType, ModelStatus, ProblemType, TrainingConfig


class ActivationType(Enum):
    """Types of activation functions."""
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"
    LINEAR = "linear"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    SWISH = "swish"


class OptimizerType(Enum):
    """Types of optimizers."""
    ADAM = "adam"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"
    ADADELTA = "adadelta"


class NetworkArchitecture(Enum):
    """Types of neural network architectures."""
    FEEDFORWARD = "feedforward"
    CONVOLUTIONAL = "convolutional"
    RECURRENT = "recurrent"
    AUTOENCODER = "autoencoder"
    CUSTOM = "custom"


@dataclass
class NeuralNetworkConfig:
    """Configuration for neural networks."""
    architecture: NetworkArchitecture
    hidden_layers: List[int] = field(default_factory=lambda: [64, 32])
    activation: ActivationType = ActivationType.RELU
    output_activation: Optional[ActivationType] = None
    optimizer: OptimizerType = OptimizerType.ADAM
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    dropout_rate: float = 0.0
    regularization_l1: float = 0.0
    regularization_l2: float = 0.0
    early_stopping: bool = True
    patience: int = 10
    use_batch_normalization: bool = False
    random_state: Optional[int] = 42


@dataclass
class TrainingHistory:
    """Training history for neural networks."""
    loss: List[float] = field(default_factory=list)
    accuracy: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    epochs_trained: int = 0
    best_epoch: Optional[int] = None
    training_time: float = 0.0


class BaseNeuralNetwork(BaseModel):
    """
    Base class for neural network models.
    
    Provides common functionality for all neural network types including
    training, prediction, visualization, and evaluation.
    """
    
    def __init__(self, 
                 metadata: ModelMetadata,
                 nn_config: NeuralNetworkConfig,
                 config: Optional[TrainingConfig] = None):
        """
        Initialize neural network model.
        
        Args:
            metadata: Model metadata
            nn_config: Neural network specific configuration
            config: General training configuration
        """
        super().__init__(metadata, config)
        self.nn_config = nn_config
        self.model = None
        self.scaler = None
        self.history = TrainingHistory()
        self.input_shape = None
        self.output_shape = None
        
        # Set random seeds for reproducibility
        if nn_config.random_state is not None:
            np.random.seed(nn_config.random_state)
            if TENSORFLOW_AVAILABLE and tf is not None:
                tf.random.set_seed(nn_config.random_state)
    
    def _build_model(self, input_shape: Tuple[int, ...], output_shape: int) -> None:
        """Build the neural network model."""
        if TENSORFLOW_AVAILABLE:
            self._build_tensorflow_model(input_shape, output_shape)
        else:
            self._build_sklearn_model()
    
    def _build_tensorflow_model(self, input_shape: Tuple[int, ...], output_shape: int) -> None:
        """Build TensorFlow/Keras model."""
        if not TENSORFLOW_AVAILABLE or models is None or layers is None or keras is None:
            raise ValueError("TensorFlow/Keras not available")
        
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=input_shape))
        
        # Hidden layers
        for i, units in enumerate(self.nn_config.hidden_layers):
            # Dense layer
            model.add(layers.Dense(
                units, 
                activation=self.nn_config.activation.value,
                kernel_regularizer=self._get_regularizer()
            ))
            
            # Batch normalization
            if self.nn_config.use_batch_normalization:
                model.add(layers.BatchNormalization())
            
            # Dropout
            if self.nn_config.dropout_rate > 0:
                model.add(layers.Dropout(self.nn_config.dropout_rate))
        
        # Output layer
        output_activation = self._get_output_activation(output_shape)
        model.add(layers.Dense(output_shape, activation=output_activation))
        
        # Compile model
        if TENSORFLOW_AVAILABLE and models is not None:
            optimizer = self._get_optimizer_string()
            loss = self._get_loss_function(output_shape)
            metrics = self._get_metrics()
            
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        self.model = model
    
    def _build_sklearn_model(self) -> None:
        """Build sklearn neural network model as fallback."""
        if self.metadata.model_type == ModelType.CLASSIFICATION:
            self.model = MLPClassifier(
                hidden_layer_sizes=tuple(self.nn_config.hidden_layers),
                activation=self._map_activation_sklearn(),
                solver='adam',
                learning_rate_init=self.nn_config.learning_rate,
                max_iter=self.nn_config.epochs,
                random_state=self.nn_config.random_state,
                early_stopping=self.nn_config.early_stopping
            )
        else:
            self.model = MLPRegressor(
                hidden_layer_sizes=tuple(self.nn_config.hidden_layers),
                activation=self._map_activation_sklearn(),
                solver='adam',
                learning_rate_init=self.nn_config.learning_rate,
                max_iter=self.nn_config.epochs,
                random_state=self.nn_config.random_state,
                early_stopping=self.nn_config.early_stopping
            )
        
        # Initialize scaler for sklearn models
        self.scaler = StandardScaler()
    
    def _get_regularizer(self):
        """Get regularization for layers."""
        if not TENSORFLOW_AVAILABLE or keras is None:
            return None
            
        l1 = self.nn_config.regularization_l1
        l2 = self.nn_config.regularization_l2
        
        if l1 > 0 and l2 > 0:
            return keras.regularizers.L1L2(l1=l1, l2=l2)
        elif l1 > 0:
            return keras.regularizers.L1(l1)
        elif l2 > 0:
            return keras.regularizers.L2(l2)
        else:
            return None
    
    def _get_output_activation(self, output_shape: int) -> str:
        """Get output activation function."""
        if self.nn_config.output_activation:
            return self.nn_config.output_activation.value
        
        # Auto-select based on problem type
        if self.metadata.model_type == ModelType.CLASSIFICATION:
            if output_shape == 1 or self.metadata.problem_type == ProblemType.BINARY_CLASSIFICATION:
                return "sigmoid"
            else:
                return "softmax"
        else:
            return "linear"
    
    def _get_optimizer(self):
        """Get optimizer for training."""
        if not TENSORFLOW_AVAILABLE or optimizers is None:
            return None
            
        lr = self.nn_config.learning_rate
        
        if self.nn_config.optimizer == OptimizerType.ADAM:
            return optimizers.Adam(learning_rate=lr)
        elif self.nn_config.optimizer == OptimizerType.SGD:
            return optimizers.SGD(learning_rate=lr)
        elif self.nn_config.optimizer == OptimizerType.RMSPROP:
            return optimizers.RMSprop(learning_rate=lr)
        elif self.nn_config.optimizer == OptimizerType.ADAGRAD:
            return optimizers.Adagrad(learning_rate=lr)
        else:
            return optimizers.Adam(learning_rate=lr)
    
    def _get_optimizer_string(self):
        """Get optimizer as string for compilation."""
        if self.nn_config.optimizer == OptimizerType.ADAM:
            return "adam"
        elif self.nn_config.optimizer == OptimizerType.SGD:
            return "sgd"
        elif self.nn_config.optimizer == OptimizerType.RMSPROP:
            return "rmsprop"
        elif self.nn_config.optimizer == OptimizerType.ADAGRAD:
            return "adagrad"
        else:
            return "adam"
    
    def _get_loss_function(self, output_shape: int) -> str:
        """Get loss function for training."""
        if self.metadata.model_type == ModelType.CLASSIFICATION:
            if output_shape == 1 or self.metadata.problem_type == ProblemType.BINARY_CLASSIFICATION:
                return "binary_crossentropy"
            else:
                return "sparse_categorical_crossentropy"
        else:
            return "mse"
    
    def _get_metrics(self) -> List[str]:
        """Get metrics for training."""
        if self.metadata.model_type == ModelType.CLASSIFICATION:
            return ["accuracy"]
        else:
            return ["mae"]
    
    def _map_activation_sklearn(self):
        """Map activation function to sklearn format."""
        if self.nn_config.activation == ActivationType.RELU:
            return "relu"
        elif self.nn_config.activation == ActivationType.SIGMOID:
            return "logistic"
        elif self.nn_config.activation == ActivationType.TANH:
            return "tanh"
        else:
            return "relu"
    
    def _prepare_callbacks(self) -> List:
        """Prepare training callbacks."""
        if not TENSORFLOW_AVAILABLE or callbacks is None:
            return []
            
        callback_list = []
        
        # Early stopping
        if self.nn_config.early_stopping:
            callback_list.append(callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.nn_config.patience,
                restore_best_weights=True,
                verbose=1
            ))
        
        # Reduce learning rate on plateau
        callback_list.append(callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ))
        
        return callback_list
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'BaseNeuralNetwork':
        """
        Train the neural network.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Fitted neural network model
        """
        start_time = time.time()
        self.update_status(ModelStatus.TRAINING)
        
        # Validate and prepare data
        X_validated, y_validated = self._validate_data(X, y)
        if X_validated is None or y_validated is None:
            raise ValueError("Data validation failed")
        
        # Scale features for sklearn models
        if not TENSORFLOW_AVAILABLE and self.scaler is not None:
            X_validated = self.scaler.fit_transform(X_validated)
        
        # Determine input and output shapes
        self.input_shape = (X_validated.shape[1],)
        
        if self.metadata.model_type == ModelType.CLASSIFICATION:
            unique_classes = np.unique(y_validated)
            self.output_shape = len(unique_classes) if len(unique_classes) > 2 else 1
        else:
            self.output_shape = 1
        
        # Build model
        self._build_model(self.input_shape, self.output_shape)
        
        # Train model
        if (TENSORFLOW_AVAILABLE and self.model is not None and 
            hasattr(self.model, 'fit') and 
            not isinstance(self.model, (MLPClassifier, MLPRegressor))):
            # TensorFlow training
            callbacks_list = self._prepare_callbacks()
            
            history = self.model.fit(
                X_validated, y_validated,
                batch_size=self.nn_config.batch_size,
                epochs=self.nn_config.epochs,
                validation_split=self.nn_config.validation_split,
                callbacks=callbacks_list
            )
            
            # Store training history
            if hasattr(history, 'history'):
                self.history.loss = history.history.get('loss', [])
                self.history.val_loss = history.history.get('val_loss', [])
                self.history.accuracy = history.history.get('accuracy', [])
                self.history.val_accuracy = history.history.get('val_accuracy', [])
                self.history.epochs_trained = len(self.history.loss)
            
        else:
            # Sklearn training
            if self.model is not None:
                self.model.fit(X_validated, y_validated)
                if hasattr(self.model, 'n_iter_'):
                    self.history.epochs_trained = self.model.n_iter_
        
        self.is_fitted = True
        
        # Update metadata
        training_time = time.time() - start_time
        self.metadata.training_time = training_time
        self.history.training_time = training_time
        self.update_status(ModelStatus.TRAINED)
        
        # Calculate training metrics
        y_pred = self.predict(X)
        if self.metadata.model_type == ModelType.CLASSIFICATION:
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(y, y_pred)
            self.add_metric('training_accuracy', float(accuracy))
        else:
            from sklearn.metrics import mean_squared_error, r2_score
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            self.add_metric('training_mse', float(mse))
            self.add_metric('training_r2', float(r2))
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions using the neural network."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.model is None:
            raise ValueError("Model not initialized")
        
        X_validated, _ = self._validate_data(X)
        if X_validated is None:
            raise ValueError("Data validation failed")
        
        # Scale features for sklearn models
        if not TENSORFLOW_AVAILABLE and self.scaler is not None:
            X_validated = self.scaler.transform(X_validated)
        
        if TENSORFLOW_AVAILABLE and hasattr(self.model, 'predict'):
            predictions = self.model.predict(X_validated)
            
            # Convert probabilities to class predictions for classification
            if self.metadata.model_type == ModelType.CLASSIFICATION:
                if self.output_shape == 1:
                    # Binary classification
                    predictions = (predictions > 0.5).astype(int).flatten()
                else:
                    # Multi-class classification
                    predictions = np.argmax(predictions, axis=1)
            else:
                predictions = predictions.flatten()
        else:
            predictions = self.model.predict(X_validated)
        
        return predictions
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Get prediction probabilities (classification only)."""
        if self.metadata.model_type != ModelType.CLASSIFICATION:
            raise ValueError("predict_proba is only available for classification")
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.model is None:
            raise ValueError("Model not initialized")
        
        X_validated, _ = self._validate_data(X)
        if X_validated is None:
            raise ValueError("Data validation failed")
        
        # Scale features for sklearn models
        if not TENSORFLOW_AVAILABLE and self.scaler is not None:
            X_validated = self.scaler.transform(X_validated)
        
        if TENSORFLOW_AVAILABLE and hasattr(self.model, 'predict'):
            probabilities = self.model.predict(X_validated)
            
            # Ensure probabilities have correct shape
            if self.output_shape == 1:
                # Binary classification - add complement probability
                probabilities = np.column_stack([1 - probabilities, probabilities])
            
            return probabilities
        else:
            # Check if sklearn model has predict_proba method and is classification
            if (hasattr(self.model, 'predict_proba') and 
                self.metadata.model_type == ModelType.CLASSIFICATION and
                isinstance(self.model, MLPClassifier)):
                return self.model.predict_proba(X_validated)
            else:
                raise ValueError("Model does not support probability predictions")
    
    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        if (TENSORFLOW_AVAILABLE and self.model is not None and 
            hasattr(self.model, 'summary') and 
            not isinstance(self.model, (MLPClassifier, MLPRegressor))):
            # Capture TensorFlow model summary
            import io
            import contextlib
            
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                self.model.summary()
            return f.getvalue()
        else:
            # Create summary for sklearn model
            summary = f"Neural Network Model Summary\n"
            summary += f"Architecture: {self.nn_config.architecture.value}\n"
            summary += f"Hidden layers: {self.nn_config.hidden_layers}\n"
            summary += f"Activation: {self.nn_config.activation.value}\n"
            summary += f"Total parameters: {self._count_parameters()}\n"
            return summary
    
    def _count_parameters(self) -> int:
        """Count total number of parameters."""
        if (TENSORFLOW_AVAILABLE and self.model is not None and 
            hasattr(self.model, 'count_params') and 
            not isinstance(self.model, (MLPClassifier, MLPRegressor))):
            return self.model.count_params()
        else:
            # Approximate for sklearn models
            total_params = 0
            if self.input_shape is not None and isinstance(self.input_shape, tuple) and len(self.input_shape) > 0:
                layers = [self.input_shape[0]] + self.nn_config.hidden_layers + [self.output_shape]
                
                for i in range(len(layers) - 1):
                    # Weights + biases
                    total_params += layers[i] * layers[i + 1] + layers[i + 1]
            
            return total_params
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """Plot training history."""
        if not self.history.loss:
            print("No training history available to plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        axes[0].plot(self.history.loss, label='Training Loss', color='blue')
        if self.history.val_loss:
            axes[0].plot(self.history.val_loss, label='Validation Loss', color='red')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot accuracy/metrics
        if self.history.accuracy:
            axes[1].plot(self.history.accuracy, label='Training Accuracy', color='blue')
            if self.history.val_accuracy:
                axes[1].plot(self.history.val_accuracy, label='Validation Accuracy', color='red')
            axes[1].set_title('Model Accuracy')
            axes[1].set_ylabel('Accuracy')
        else:
            axes[1].text(0.5, 0.5, 'No accuracy metrics available', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Metrics')
        
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_architecture(self, save_path: Optional[str] = None) -> None:
        """Plot model architecture (TensorFlow only)."""
        if TENSORFLOW_AVAILABLE and plot_model is not None and self.model is not None and hasattr(self.model, 'summary'):
            try:
                plot_model(
                    self.model, 
                    to_file=save_path or 'model_architecture.png',
                    show_shapes=True,
                    show_layer_names=True,
                    rankdir='TB'
                )
                print(f"Model architecture saved to {save_path or 'model_architecture.png'}")
            except Exception as e:
                print(f"Could not plot model architecture: {e}")
        else:
            print("Model architecture plotting only available with TensorFlow")


class FeedforwardNetwork(BaseNeuralNetwork):
    """
    Feedforward Neural Network (Multi-layer Perceptron).
    
    A standard fully-connected neural network for classification and regression.
    """
    
    def __init__(self, 
                 name: str = "Feedforward Neural Network",
                 description: str = "Multi-layer perceptron for classification and regression",
                 problem_type: ProblemType = ProblemType.BINARY_CLASSIFICATION,
                 hidden_layers: Optional[List[int]] = None,
                 activation: ActivationType = ActivationType.RELU,
                 **kwargs):
        """
        Initialize feedforward neural network.
        
        Args:
            name: Model name
            description: Model description
            problem_type: Type of problem
            hidden_layers: List of hidden layer sizes
            activation: Activation function
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
                'hidden_layers': hidden_layers or [64, 32],
                'activation': activation.value,
                **kwargs
            }
        )
        
        # Create neural network configuration
        nn_config = NeuralNetworkConfig(
            architecture=NetworkArchitecture.FEEDFORWARD,
            hidden_layers=hidden_layers or [64, 32],
            activation=activation,
            **kwargs
        )
        
        super().__init__(metadata, nn_config)


class ConvolutionalNetwork(BaseNeuralNetwork):
    """
    Convolutional Neural Network for image data.
    
    Specialized for processing grid-like data such as images.
    """
    
    def __init__(self, 
                 name: str = "Convolutional Neural Network",
                 description: str = "CNN for image classification and regression",
                 problem_type: ProblemType = ProblemType.MULTICLASS_CLASSIFICATION,
                 conv_layers: Optional[List[Dict]] = None,
                 **kwargs):
        """
        Initialize convolutional neural network.
        
        Args:
            name: Model name
            description: Model description
            problem_type: Type of problem
            conv_layers: List of convolutional layer configurations
            **kwargs: Additional configuration parameters
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
                'conv_layers': conv_layers or [
                    {'filters': 32, 'kernel_size': 3},
                    {'filters': 64, 'kernel_size': 3}
                ],
                **kwargs
            }
        )
        
        nn_config = NeuralNetworkConfig(
            architecture=NetworkArchitecture.CONVOLUTIONAL,
            **kwargs
        )
        
        super().__init__(metadata, nn_config)
        self.conv_layers = conv_layers or [
            {'filters': 32, 'kernel_size': 3},
            {'filters': 64, 'kernel_size': 3}
        ]
    
    def _build_tensorflow_model(self, input_shape: Tuple[int, ...], output_shape: int) -> None:
        """Build CNN model for TensorFlow."""
        if not TENSORFLOW_AVAILABLE or models is None or layers is None:
            raise ValueError("TensorFlow required for CNN models")
        
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=input_shape))
        
        # Convolutional layers
        for i, conv_config in enumerate(self.conv_layers):
            model.add(layers.Conv2D(
                filters=conv_config['filters'],
                kernel_size=conv_config['kernel_size'],
                activation=self.nn_config.activation.value,
                padding='same'
            ))
            
            if self.nn_config.use_batch_normalization:
                model.add(layers.BatchNormalization())
            
            model.add(layers.MaxPooling2D(pool_size=2))
            
            if self.nn_config.dropout_rate > 0:
                model.add(layers.Dropout(self.nn_config.dropout_rate))
        
        # Flatten and dense layers
        model.add(layers.Flatten())
        
        for units in self.nn_config.hidden_layers:
            model.add(layers.Dense(
                units,
                activation=self.nn_config.activation.value,
                kernel_regularizer=self._get_regularizer()
            ))
            
            if self.nn_config.dropout_rate > 0:
                model.add(layers.Dropout(self.nn_config.dropout_rate))
        
        # Output layer
        output_activation = self._get_output_activation(output_shape)
        model.add(layers.Dense(output_shape, activation=output_activation))
        
        # Compile model
        optimizer = self._get_optimizer_string()
        loss = self._get_loss_function(output_shape)
        metrics = self._get_metrics()
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        self.model = model


# Factory functions for easy model creation
def create_feedforward_network(problem_type: ProblemType = ProblemType.BINARY_CLASSIFICATION,
                              hidden_layers: Optional[List[int]] = None,
                              activation: ActivationType = ActivationType.RELU,
                              **kwargs) -> FeedforwardNetwork:
    """Create a feedforward neural network."""
    return FeedforwardNetwork(
        problem_type=problem_type,
        hidden_layers=hidden_layers,
        activation=activation,
        **kwargs
    )


def create_cnn(problem_type: ProblemType = ProblemType.MULTICLASS_CLASSIFICATION,
               conv_layers: Optional[List[Dict]] = None,
               **kwargs) -> ConvolutionalNetwork:
    """Create a convolutional neural network."""
    return ConvolutionalNetwork(
        problem_type=problem_type,
        conv_layers=conv_layers,
        **kwargs
    )


def create_mlp_classifier(hidden_layers: Optional[List[int]] = None,
                         activation: ActivationType = ActivationType.RELU,
                         **kwargs) -> FeedforwardNetwork:
    """Create a multi-layer perceptron for classification."""
    return FeedforwardNetwork(
        name="MLP Classifier",
        problem_type=ProblemType.MULTICLASS_CLASSIFICATION,
        hidden_layers=hidden_layers or [100, 50],
        activation=activation,
        **kwargs
    )


def create_mlp_regressor(hidden_layers: Optional[List[int]] = None,
                        activation: ActivationType = ActivationType.RELU,
                        **kwargs) -> FeedforwardNetwork:
    """Create a multi-layer perceptron for regression."""
    return FeedforwardNetwork(
        name="MLP Regressor",
        problem_type=ProblemType.LINEAR_REGRESSION,
        hidden_layers=hidden_layers or [100, 50],
        activation=activation,
        **kwargs
    )


# Export main classes and functions
__all__ = [
    # Base classes
    'BaseNeuralNetwork',
    'ActivationType',
    'OptimizerType',
    'NetworkArchitecture',
    'NeuralNetworkConfig',
    'TrainingHistory',
    
    # Neural network models
    'FeedforwardNetwork',
    'ConvolutionalNetwork',
    
    # Factory functions
    'create_feedforward_network',
    'create_cnn',
    'create_mlp_classifier',
    'create_mlp_regressor'
]
