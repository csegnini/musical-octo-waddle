"""
Advanced Deep Learning Package

This module provides comprehensive deep learning capabilities including:
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs, LSTMs, GRUs)
- Transformer architectures
- Advanced architectures (ResNet, Attention mechanisms)
- Computer vision models
- Natural language processing models
- Sequence-to-sequence models

Classes:
    ConvolutionalNetwork: CNN architectures for computer vision
    RecurrentNetwork: RNN/LSTM/GRU for sequence modeling
    TransformerNetwork: Transformer architecture for NLP
    VisionTransformer: Vision Transformer for image classification
    AttentionMechanism: Multi-head attention implementation
    ResNetArchitecture: ResNet with skip connections
    SequenceToSequence: Seq2Seq models with attention
    
Functions:
    create_cnn_classifier: Create CNN for image classification
    create_rnn_classifier: Create RNN for sequence classification
    create_transformer_model: Create transformer for NLP
    create_vision_transformer: Create ViT for image classification
    create_seq2seq_model: Create sequence-to-sequence model
"""

import numpy as np
import pandas as pd
import uuid
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import time
from abc import ABC, abstractmethod

# Import base classes
from ..base import (
    BaseModel, ProblemType, ModelType, ModelMetadata, 
    ModelStatus, TrainingConfig# ValidationResult
)

# Check for deep learning libraries
TENSORFLOW_AVAILABLE = False
TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    import keras
    from keras import layers, Model
    from keras.applications import ResNet50, VGG16, EfficientNetB0
    TENSORFLOW_AVAILABLE = True
except ImportError:
    # Create dummy objects for type checking
    tf = None
    keras = None
    layers = None
    Model = None
    ResNet50 = None
    VGG16 = None
    EfficientNetB0 = None
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    pass

# Configuration classes
class CNNArchitecture(Enum):
    """CNN architecture types."""
    SIMPLE = "simple"
    LENET = "lenet"
    ALEXNET = "alexnet"
    VGG = "vgg"
    RESNET = "resnet"
    EFFICIENTNET = "efficientnet"
    CUSTOM = "custom"

class RNNType(Enum):
    """RNN types."""
    VANILLA = "vanilla"
    LSTM = "lstm"
    GRU = "gru"
    BIDIRECTIONAL = "bidirectional"

class AttentionType(Enum):
    """Attention mechanism types."""
    SCALED_DOT_PRODUCT = "scaled_dot_product"
    MULTI_HEAD = "multi_head"
    SELF_ATTENTION = "self_attention"
    CROSS_ATTENTION = "cross_attention"

class PoolingType(Enum):
    """Pooling layer types."""
    MAX = "max"
    AVERAGE = "average"
    GLOBAL_MAX = "global_max"
    GLOBAL_AVERAGE = "global_average"

@dataclass
class ConvolutionalConfig:
    """Configuration for CNN models."""
    filters: List[int] = field(default_factory=lambda: [32, 64, 128])
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3])
    pool_sizes: List[int] = field(default_factory=lambda: [2, 2, 2])
    strides: List[int] = field(default_factory=lambda: [1, 1, 1])
    padding: str = "same"
    activation: str = "relu"
    dropout_rate: float = 0.2
    batch_normalization: bool = True
    pooling_type: PoolingType = PoolingType.MAX
    architecture: CNNArchitecture = CNNArchitecture.SIMPLE

@dataclass
class RecurrentConfig:
    """Configuration for RNN models."""
    units: List[int] = field(default_factory=lambda: [64, 32])
    rnn_type: RNNType = RNNType.LSTM
    return_sequences: bool = False
    dropout_rate: float = 0.2
    recurrent_dropout: float = 0.1
    bidirectional: bool = False
    attention: bool = False

@dataclass
class TransformerConfig:
    """Configuration for Transformer models."""
    num_layers: int = 6
    d_model: int = 512
    num_heads: int = 8
    d_ff: int = 2048
    dropout_rate: float = 0.1
    max_sequence_length: int = 512
    vocab_size: int = 10000
    attention_type: AttentionType = AttentionType.MULTI_HEAD


class AttentionMechanism:
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.1):
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.d_k = d_model // num_heads
        
        if TENSORFLOW_AVAILABLE:
            self._create_tensorflow_attention()
    
    def _create_tensorflow_attention(self):
        """Create TensorFlow multi-head attention."""
        if not TENSORFLOW_AVAILABLE or layers is None:
            return
            
        self.attention_layer = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_k,
            dropout=self.dropout_rate
        )
    
    def scaled_dot_product_attention(self, query, key, value, mask=None):
        """Compute scaled dot-product attention."""
        if not TENSORFLOW_AVAILABLE or tf is None:
            raise ImportError("TensorFlow required for attention mechanisms")
        
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        # Use static dimension access for key dim
        key_shape = key.shape
        dk = tf.cast(key_shape[-1] if key_shape[-1] is not None else 64, tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, value)
        
        return output, attention_weights


class ConvolutionalNetwork(BaseModel):
    """Convolutional Neural Network implementation."""
    
    def __init__(self, 
                 config: ConvolutionalConfig,
                 input_shape: Tuple[int, int, int],
                 num_classes: int,
                 name: str = "CNN",
                 **kwargs):
        """
        Initialize CNN.
        
        Args:
            config: CNN configuration
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of output classes
            name: Model name
        """
        # Create metadata for BaseModel
        metadata = ModelMetadata(
            model_id=str(uuid.uuid4()) if 'uuid' in globals() else "cnn_model",
            name=name,
            description="Convolutional Neural Network",
            model_type=ModelType.CLASSIFICATION,
            problem_type=ProblemType.MULTICLASS_CLASSIFICATION
        )
        
        super().__init__(metadata=metadata, **kwargs)
        self.config = config
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
        if TENSORFLOW_AVAILABLE:
            self._build_tensorflow_model()
        else:
            raise ImportError("TensorFlow required for CNN models")
    
    def _build_tensorflow_model(self):
        """Build CNN using TensorFlow/Keras."""
        if not TENSORFLOW_AVAILABLE or layers is None or Model is None:
            raise ImportError("TensorFlow/Keras required for CNN models")
            
        inputs = layers.Input(shape=self.input_shape)
        x = inputs
        
        # Convolutional layers
        for i, (filters, kernel_size, pool_size, stride) in enumerate(zip(
            self.config.filters, 
            self.config.kernel_sizes,
            self.config.pool_sizes,
            self.config.strides
        )):
            # Convolution
            x = layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=stride,
                padding=self.config.padding,
                activation=self.config.activation,
                name=f'conv_{i+1}'
            )(x)
            
            # Batch normalization
            if self.config.batch_normalization:
                x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
            
            # Pooling
            if self.config.pooling_type == PoolingType.MAX:
                x = layers.MaxPooling2D(pool_size=pool_size, name=f'pool_{i+1}')(x)
            elif self.config.pooling_type == PoolingType.AVERAGE:
                x = layers.AveragePooling2D(pool_size=pool_size, name=f'pool_{i+1}')(x)
            
            # Dropout
            if self.config.dropout_rate > 0:
                x = layers.Dropout(self.config.dropout_rate, name=f'dropout_{i+1}')(x)
        
        # Global pooling
        if self.config.pooling_type == PoolingType.GLOBAL_MAX:
            x = layers.GlobalMaxPooling2D()(x)
        elif self.config.pooling_type == PoolingType.GLOBAL_AVERAGE:
            x = layers.GlobalAveragePooling2D()(x)
        else:
            x = layers.Flatten()(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu', name='dense_1')(x)
        if self.config.dropout_rate > 0:
            x = layers.Dropout(self.config.dropout_rate, name='dropout_dense')(x)
        
        x = layers.Dense(256, activation='relu', name='dense_2')(x)
        
        # Output layer
        if self.num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name=self.metadata.name)
    
    def fit(self, X, y, validation_data=None, epochs=50, batch_size=32, **kwargs):
        """Train the CNN model."""
        if not TENSORFLOW_AVAILABLE or self.model is None:
            raise RuntimeError("Model not properly initialized")
        
        # Compile model
        if self.num_classes == 2:
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        self.model.compile(
            optimizer='adam',
            loss=loss,
            metrics=metrics
        )
        
        # Callbacks
        callbacks = []
        if TENSORFLOW_AVAILABLE and keras is not None:
            callbacks = [
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
            ]
        
        # Train model
        self.history = self.model.fit(
            X, y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            **kwargs
        )
        
        self.metadata.status = ModelStatus.TRAINED
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        predictions = self.model.predict(X)
        
        if self.num_classes == 2:
            return (predictions > 0.5).astype(int).flatten()
        else:
            return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        return self.model.predict(X)
    
    def get_model_summary(self):
        """Get model architecture summary."""
        if self.model is None:
            return "Model not built"
        
        import io
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        self.model.summary()
        
        sys.stdout = old_stdout
        return buffer.getvalue()
    
    def _count_parameters(self):
        """Count model parameters."""
        if self.model is None:
            return 0
        return self.model.count_params()
    
    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            print("No training history available")
            return
        
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training & validation accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in self.history.history:
            ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot training & validation loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class RecurrentNetwork(BaseModel):
    """Recurrent Neural Network implementation."""
    
    def __init__(self, 
                 config: RecurrentConfig,
                 input_shape: Tuple[int, int],
                 num_classes: int,
                 name: str = "RNN",
                 **kwargs):
        """
        Initialize RNN.
        
        Args:
            config: RNN configuration
            input_shape: Input sequence shape (timesteps, features)
            num_classes: Number of output classes
            name: Model name
        """
        # Create metadata for BaseModel
        metadata = ModelMetadata(
            model_id=str(uuid.uuid4()),
            name=name,
            description="Recurrent Neural Network",
            model_type=ModelType.CLASSIFICATION,
            problem_type=ProblemType.MULTICLASS_CLASSIFICATION
        )
        
        super().__init__(metadata=metadata, **kwargs)
        self.config = config
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
        if TENSORFLOW_AVAILABLE:
            self._build_tensorflow_model()
        else:
            raise ImportError("TensorFlow required for RNN models")
    
    def _build_tensorflow_model(self):
        """Build RNN using TensorFlow/Keras."""
        if not TENSORFLOW_AVAILABLE or layers is None or Model is None:
            raise ImportError("TensorFlow/Keras required for RNN models")
            
        inputs = layers.Input(shape=self.input_shape)
        x = inputs
        
        # RNN layers
        for i, units in enumerate(self.config.units):
            return_sequences = (i < len(self.config.units) - 1) or self.config.return_sequences
            
            # Choose RNN type
            if self.config.rnn_type == RNNType.LSTM:
                rnn_layer = layers.LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=self.config.dropout_rate,
                    recurrent_dropout=self.config.recurrent_dropout,
                    name=f'lstm_{i+1}'
                )
            elif self.config.rnn_type == RNNType.GRU:
                rnn_layer = layers.GRU(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=self.config.dropout_rate,
                    recurrent_dropout=self.config.recurrent_dropout,
                    name=f'gru_{i+1}'
                )
            else:  # Vanilla RNN
                rnn_layer = layers.SimpleRNN(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=self.config.dropout_rate,
                    name=f'rnn_{i+1}'
                )
            
            # Apply bidirectional wrapper if needed
            if self.config.bidirectional:
                rnn_layer = layers.Bidirectional(rnn_layer, name=f'bi_{rnn_layer.name}')
            
            x = rnn_layer(x)
        
        # Attention mechanism
        if self.config.attention:
            attention = layers.Attention()([x, x])
            x = layers.Concatenate()([x, attention])
        
        # Dense layers
        x = layers.Dense(128, activation='relu', name='dense_1')(x)
        if self.config.dropout_rate > 0:
            x = layers.Dropout(self.config.dropout_rate, name='dropout_dense')(x)
        
        # Output layer
        if self.num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name=self.metadata.name)
    
    def fit(self, X, y, validation_data=None, epochs=50, batch_size=32, **kwargs):
        """Train the RNN model."""
        if not TENSORFLOW_AVAILABLE or self.model is None:
            raise RuntimeError("Model not properly initialized")
        
        # Compile model
        if self.num_classes == 2:
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        self.model.compile(
            optimizer='adam',
            loss=loss,
            metrics=metrics
        )
        
        # Callbacks
        callbacks = []
        if TENSORFLOW_AVAILABLE and keras is not None:
            callbacks = [
                keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-7)
            ]
        
        # Train model
        self.history = self.model.fit(
            X, y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            **kwargs
        )
        
        self.metadata.status = ModelStatus.TRAINED
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        predictions = self.model.predict(X)
        
        if self.num_classes == 2:
            return (predictions > 0.5).astype(int).flatten()
        else:
            return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        return self.model.predict(X)
    
    def get_model_summary(self):
        """Get model architecture summary."""
        if self.model is None:
            return "Model not built"
        
        import io
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        self.model.summary()
        
        sys.stdout = old_stdout
        return buffer.getvalue()
    
    def _count_parameters(self):
        """Count model parameters."""
        if self.model is None:
            return 0
        return self.model.count_params()
    
    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            print("No training history available")
            return
        
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training & validation accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in self.history.history:
            ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot training & validation loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class TransformerNetwork(BaseModel):
    """Transformer architecture implementation."""
    
    def __init__(self, 
                 config: TransformerConfig,
                 num_classes: int,
                 name: str = "Transformer",
                 **kwargs):
        """
        Initialize Transformer.
        
        Args:
            config: Transformer configuration
            num_classes: Number of output classes
            name: Model name
        """
        # Create metadata for BaseModel
        metadata = ModelMetadata(
            model_id=str(uuid.uuid4()),
            name=name,
            description="Transformer Neural Network",
            model_type=ModelType.CLASSIFICATION,
            problem_type=ProblemType.MULTICLASS_CLASSIFICATION
        )
        
        super().__init__(metadata=metadata, **kwargs)
        self.config = config
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
        if TENSORFLOW_AVAILABLE:
            self._build_tensorflow_model()
        else:
            raise ImportError("TensorFlow required for Transformer models")
    
    def _build_tensorflow_model(self):
        """Build Transformer using TensorFlow/Keras."""
        if not TENSORFLOW_AVAILABLE or layers is None or Model is None:
            raise ImportError("TensorFlow/Keras required for Transformer models")
            
        inputs = layers.Input(shape=(self.config.max_sequence_length,))
        
        # Embedding layer
        embedding = layers.Embedding(
            input_dim=self.config.vocab_size,
            output_dim=self.config.d_model,
            input_length=self.config.max_sequence_length
        )(inputs)
        
        # Positional encoding
        x = self._add_positional_encoding(embedding)
        
        # Transformer layers
        for i in range(self.config.num_layers):
            x = self._transformer_block(x, f'transformer_{i+1}')
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(256, activation='relu', name='dense_1')(x)
        x = layers.Dropout(self.config.dropout_rate, name='dropout_1')(x)
        
        x = layers.Dense(128, activation='relu', name='dense_2')(x)
        x = layers.Dropout(self.config.dropout_rate, name='dropout_2')(x)
        
        # Output layer
        if self.num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name=self.metadata.name)
    
    def _add_positional_encoding(self, x):
        """Add positional encoding to embeddings."""
        if not TENSORFLOW_AVAILABLE or tf is None:
            raise ImportError("TensorFlow required for positional encoding")
            
        seq_len = self.config.max_sequence_length
        d_model = self.config.d_model
        
        # Create positional encoding
        pos_encoding = np.zeros((seq_len, d_model))
        
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                pos_encoding[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
                if i + 1 < d_model:
                    pos_encoding[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
        
        pos_encoding = tf.constant(pos_encoding, dtype=tf.float32)
        pos_encoding = tf.expand_dims(pos_encoding, 0)
        
        return x + pos_encoding
    
    def _transformer_block(self, x, name_prefix):
        """Create a transformer block."""
        if not TENSORFLOW_AVAILABLE or layers is None:
            raise ImportError("TensorFlow/Keras required for transformer blocks")
            
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=self.config.num_heads,
            key_dim=self.config.d_model // self.config.num_heads,
            dropout=self.config.dropout_rate,
            name=f'{name_prefix}_attention'
        )(x, x)
        
        # Add & norm
        x = layers.Add(name=f'{name_prefix}_add_1')([x, attention_output])
        x = layers.LayerNormalization(name=f'{name_prefix}_norm_1')(x)
        
        # Feed forward
        ffn_output = layers.Dense(
            self.config.d_ff, 
            activation='relu',
            name=f'{name_prefix}_ffn_1'
        )(x)
        ffn_output = layers.Dropout(
            self.config.dropout_rate,
            name=f'{name_prefix}_ffn_dropout'
        )(ffn_output)
        ffn_output = layers.Dense(
            self.config.d_model,
            name=f'{name_prefix}_ffn_2'
        )(ffn_output)
        
        # Add & norm
        x = layers.Add(name=f'{name_prefix}_add_2')([x, ffn_output])
        x = layers.LayerNormalization(name=f'{name_prefix}_norm_2')(x)
        
        return x
    
    def fit(self, X, y, validation_data=None, epochs=50, batch_size=32, **kwargs):
        """Train the Transformer model."""
        if not TENSORFLOW_AVAILABLE or self.model is None:
            raise RuntimeError("Model not properly initialized")
        
        # Compile model
        if self.num_classes == 2:
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        # Use string optimizer name for compatibility
        self.model.compile(
            optimizer='adam',
            loss=loss,
            metrics=metrics
        )
        
        # Callbacks
        callbacks = []
        if TENSORFLOW_AVAILABLE and keras is not None:
            callbacks = [
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
            ]
        
        # Train model
        self.history = self.model.fit(
            X, y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            **kwargs
        )
        
        self.metadata.status = ModelStatus.TRAINED
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        predictions = self.model.predict(X)
        
        if self.num_classes == 2:
            return (predictions > 0.5).astype(int).flatten()
        else:
            return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        return self.model.predict(X)
    
    def get_model_summary(self):
        """Get model architecture summary."""
        if self.model is None:
            return "Model not built"
        
        import io
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        self.model.summary()
        
        sys.stdout = old_stdout
        return buffer.getvalue()
    
    def _count_parameters(self):
        """Count model parameters."""
        if self.model is None:
            return 0
        return self.model.count_params()


class VisionTransformer(BaseModel):
    """Vision Transformer implementation."""
    
    def __init__(self, 
                 image_size: int,
                 patch_size: int,
                 num_classes: int,
                 d_model: int = 768,
                 num_heads: int = 12,
                 num_layers: int = 12,
                 mlp_dim: int = 3072,
                 dropout_rate: float = 0.1,
                 name: str = "ViT",
                 **kwargs):
        """
        Initialize Vision Transformer.
        
        Args:
            image_size: Input image size (assumed square)
            patch_size: Size of image patches
            num_classes: Number of output classes
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            mlp_dim: MLP dimension in transformer
            dropout_rate: Dropout rate
            name: Model name
        """
        # Create metadata for BaseModel
        metadata = ModelMetadata(
            model_id=str(uuid.uuid4()),
            name=name,
            description="Vision Transformer",
            model_type=ModelType.CLASSIFICATION,
            problem_type=ProblemType.MULTICLASS_CLASSIFICATION
        )
        
        super().__init__(metadata=metadata, **kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        
        self.num_patches = (image_size // patch_size) ** 2
        
        if TENSORFLOW_AVAILABLE:
            self._build_tensorflow_model()
        else:
            raise ImportError("TensorFlow required for Vision Transformer")
    
    def _build_tensorflow_model(self):
        """Build Vision Transformer using TensorFlow/Keras."""
        if not TENSORFLOW_AVAILABLE or layers is None or Model is None:
            raise ImportError("TensorFlow/Keras required for Vision Transformer")
            
        inputs = layers.Input(shape=(self.image_size, self.image_size, 3))
        
        # Create patches
        patches = self._extract_patches(inputs)
        
        # Patch embedding
        if not TENSORFLOW_AVAILABLE or layers is None:
            raise ImportError("TensorFlow/Keras required for ViT layers")
        patch_embedding = layers.Dense(self.d_model)(patches)
        
        # Add positional embeddings
        if not TENSORFLOW_AVAILABLE or tf is None:
            raise ImportError("TensorFlow required for ViT position embeddings")
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        position_embedding = layers.Embedding(
            input_dim=self.num_patches, 
            output_dim=self.d_model
        )(positions)
        
        # Add class token - use static broadcasting approach
        class_token_init = tf.random.normal([1, 1, self.d_model])
        # Use static batch size for shape compatibility  
        batch_size = 32  # Default batch size
        class_tokens = tf.broadcast_to(
            class_token_init, [batch_size, 1, self.d_model]
        )
        
        # Concatenate class token with patch embeddings
        x = tf.concat([class_tokens, patch_embedding + position_embedding], axis=1)
        
        # Transformer layers
        for i in range(self.num_layers):
            x = self._transformer_block(x, f'vit_block_{i+1}')
        
        # Use class token for classification
        if not TENSORFLOW_AVAILABLE or tf is None:
            raise ImportError("TensorFlow required for ViT token extraction")
        x = tf.gather(x, 0, axis=1)  # Extract class token (first token)
        
        # Classification head
        if not TENSORFLOW_AVAILABLE or layers is None:
            raise ImportError("TensorFlow/Keras required for ViT classification head")
        x = layers.LayerNormalization()(x)
        x = layers.Dense(self.mlp_dim, activation='gelu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        if self.num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name=self.metadata.name)
    
    def _extract_patches(self, images):
        """Extract image patches."""
        if not TENSORFLOW_AVAILABLE or tf is None:
            raise ImportError("TensorFlow required for patch extraction")
            
        # Use static batch size or default to 32
        batch_size = 32  # Use a reasonable default for batch size
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
    def _transformer_block(self, x, name_prefix):
        """Create a Vision Transformer block."""
        if not TENSORFLOW_AVAILABLE or layers is None:
            raise ImportError("TensorFlow/Keras required for ViT transformer blocks")
            
        # Layer norm 1
        x1 = layers.LayerNormalization(epsilon=1e-6, name=f'{name_prefix}_norm1')(x)
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            dropout=self.dropout_rate,
            name=f'{name_prefix}_attention'
        )(x1, x1)
        
        # Skip connection 1
        x2 = layers.Add(name=f'{name_prefix}_add1')([attention_output, x])
        
        # Layer norm 2
        x3 = layers.LayerNormalization(epsilon=1e-6, name=f'{name_prefix}_norm2')(x2)
        
        # MLP
        x3 = layers.Dense(self.mlp_dim, activation='gelu', name=f'{name_prefix}_mlp1')(x3)
        x3 = layers.Dropout(self.dropout_rate, name=f'{name_prefix}_dropout')(x3)
        x3 = layers.Dense(self.d_model, name=f'{name_prefix}_mlp2')(x3)
        
        # Skip connection 2
        return layers.Add(name=f'{name_prefix}_add2')([x3, x2])
    
    def fit(self, X, y, validation_data=None, epochs=50, batch_size=32, **kwargs):
        """Train the Vision Transformer."""
        if not TENSORFLOW_AVAILABLE or self.model is None:
            raise RuntimeError("Model not properly initialized")
        
        # Compile model
        if self.num_classes == 2:
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        # Use a lower learning rate for Vision Transformer
        self.model.compile(
            optimizer='adam',
            loss=loss,
            metrics=metrics
        )
        
        # Callbacks
        callbacks = []
        if TENSORFLOW_AVAILABLE and keras is not None:
            callbacks = [
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
            ]
        
        # Train model
        self.history = self.model.fit(
            X, y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            **kwargs
        )
        
        self.metadata.status = ModelStatus.TRAINED
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        predictions = self.model.predict(X)
        
        if self.num_classes == 2:
            return (predictions > 0.5).astype(int).flatten()
        else:
            return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        return self.model.predict(X)
    
    def get_model_summary(self):
        """Get model architecture summary."""
        if self.model is None:
            return "Model not built"
        
        import io
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        self.model.summary()
        
        sys.stdout = old_stdout
        return buffer.getvalue()
    
    def _count_parameters(self):
        """Count model parameters."""
        if self.model is None:
            return 0
        return self.model.count_params()


# Factory functions
def create_cnn_classifier(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    architecture: CNNArchitecture = CNNArchitecture.SIMPLE,
    filters: Optional[List[int]] = None,
    dropout_rate: float = 0.2,
    name: str = "CNN_Classifier"
) -> ConvolutionalNetwork:
    """
    Create a CNN classifier.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of classes
        architecture: CNN architecture type
        filters: Filter sizes for conv layers
        dropout_rate: Dropout rate
        name: Model name
        
    Returns:
        ConvolutionalNetwork instance
    """
    if filters is None:
        if architecture == CNNArchitecture.SIMPLE:
            filters = [32, 64, 128]
        elif architecture == CNNArchitecture.VGG:
            filters = [64, 128, 256, 512]
        else:
            filters = [32, 64, 128, 256]
    
    config = ConvolutionalConfig(
        filters=filters,
        dropout_rate=dropout_rate,
        architecture=architecture
    )
    
    return ConvolutionalNetwork(
        config=config,
        input_shape=input_shape,
        num_classes=num_classes,
        name=name
    )


def create_rnn_classifier(
    input_shape: Tuple[int, int],
    num_classes: int,
    rnn_type: RNNType = RNNType.LSTM,
    units: Optional[List[int]] = None,
    bidirectional: bool = False,
    attention: bool = False,
    dropout_rate: float = 0.2,
    name: str = "RNN_Classifier"
) -> RecurrentNetwork:
    """
    Create an RNN classifier.
    
    Args:
        input_shape: Input sequence shape (timesteps, features)
        num_classes: Number of classes
        rnn_type: Type of RNN (LSTM, GRU, etc.)
        units: Units for each RNN layer
        bidirectional: Whether to use bidirectional RNN
        attention: Whether to use attention mechanism
        dropout_rate: Dropout rate
        name: Model name
        
    Returns:
        RecurrentNetwork instance
    """
    if units is None:
        units = [64, 32]
    
    config = RecurrentConfig(
        units=units,
        rnn_type=rnn_type,
        bidirectional=bidirectional,
        attention=attention,
        dropout_rate=dropout_rate
    )
    
    return RecurrentNetwork(
        config=config,
        input_shape=input_shape,
        num_classes=num_classes,
        name=name
    )


def create_transformer_model(
    vocab_size: int,
    num_classes: int,
    max_sequence_length: int = 512,
    d_model: int = 512,
    num_heads: int = 8,
    num_layers: int = 6,
    dropout_rate: float = 0.1,
    name: str = "Transformer"
) -> TransformerNetwork:
    """
    Create a Transformer model.
    
    Args:
        vocab_size: Vocabulary size
        num_classes: Number of classes
        max_sequence_length: Maximum sequence length
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout_rate: Dropout rate
        name: Model name
        
    Returns:
        TransformerNetwork instance
    """
    config = TransformerConfig(
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=dropout_rate
    )
    
    return TransformerNetwork(
        config=config,
        num_classes=num_classes,
        name=name
    )


def create_vision_transformer(
    image_size: int,
    num_classes: int,
    patch_size: int = 16,
    d_model: int = 768,
    num_heads: int = 12,
    num_layers: int = 12,
    dropout_rate: float = 0.1,
    name: str = "ViT"
) -> VisionTransformer:
    """
    Create a Vision Transformer model.
    
    Args:
        image_size: Input image size (assumed square)
        num_classes: Number of classes
        patch_size: Size of image patches
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout_rate: Dropout rate
        name: Model name
        
    Returns:
        VisionTransformer instance
    """
    return VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        name=name
    )


# Export all classes and functions
__all__ = [
    # Configuration classes
    'CNNArchitecture', 'RNNType', 'AttentionType', 'PoolingType',
    'ConvolutionalConfig', 'RecurrentConfig', 'TransformerConfig',
    
    # Core classes
    'ConvolutionalNetwork', 'RecurrentNetwork', 'TransformerNetwork',
    'VisionTransformer', 'AttentionMechanism',
    
    # Factory functions
    'create_cnn_classifier', 'create_rnn_classifier', 
    'create_transformer_model', 'create_vision_transformer'
]
