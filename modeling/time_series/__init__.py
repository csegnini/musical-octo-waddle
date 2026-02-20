"""
Time Series Analysis Module

This module provides comprehensive time series analysis capabilities including:
- Classic time series models (ARIMA, SARIMA, Exponential Smoothing)
- Trend and seasonality decomposition
- Advanced forecasting models (Prophet, LSTM)
- Time series preprocessing and feature engineering
- Model evaluation and validation
- Interactive visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Core time series libraries
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

# Statistical models
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Prophet for advanced forecasting
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# Deep learning for time series
try:
    import tensorflow as tf
    import keras
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    from keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Base classes
from ..base import BaseModel, ModelMetadata, ModelType, ProblemType, ModelStatus
from enum import Enum
from typing import Optional, Dict, List, Tuple, Any, Union
from dataclasses import dataclass
import time
import uuid


class TimeSeriesModel(Enum):
    """Supported time series models."""
    ARIMA = "arima"
    SARIMA = "sarima"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    PROPHET = "prophet"
    LSTM = "lstm"
    LINEAR_TREND = "linear_trend"
    SEASONAL_NAIVE = "seasonal_naive"


class SeasonalityType(Enum):
    """Types of seasonality patterns."""
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
    AUTO = "auto"


class TrendType(Enum):
    """Types of trend patterns."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    DAMPED = "damped"
    AUTO = "auto"


@dataclass
class TimeSeriesConfig:
    """Configuration for time series models."""
    model_type: TimeSeriesModel = TimeSeriesModel.ARIMA
    
    # ARIMA/SARIMA parameters
    order: Tuple[int, int, int] = (1, 1, 1)  # (p, d, q)
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12)  # (P, D, Q, s)
    
    # Exponential Smoothing parameters
    trend: Optional[str] = "add"
    seasonal: Optional[str] = "add"
    seasonal_periods: int = 12
    damped_trend: bool = False
    
    # Prophet parameters
    yearly_seasonality: bool = True
    weekly_seasonality: bool = True
    daily_seasonality: bool = False
    seasonality_mode: str = "additive"
    changepoint_prior_scale: float = 0.05
    seasonality_prior_scale: float = 10.0
    
    # LSTM parameters
    lookback_window: int = 60
    lstm_units: int = 50
    dropout_rate: float = 0.2
    epochs: int = 50
    batch_size: int = 32
    
    # General parameters
    random_state: int = 42


@dataclass
class ForecastResult:
    """Container for forecast results."""
    forecast: np.ndarray
    confidence_intervals: Optional[np.ndarray] = None
    model_fit: Any = None
    metrics: Optional[Dict[str, float]] = None
    forecast_dates: Optional[pd.DatetimeIndex] = None


class TimeSeriesAnalyzer(BaseModel):
    """Advanced time series analysis and forecasting model."""
    
    def __init__(self, config: TimeSeriesConfig, name: str = ""):
        """Initialize time series analyzer."""
        metadata = ModelMetadata(
            model_id=str(uuid.uuid4()),
            name=name if name else f"TimeSeries_{config.model_type.value}",
            description=f"Time series model of type {config.model_type.value}",
            model_type=ModelType.TIME_SERIES,
            problem_type=ProblemType.TIME_SERIES_FORECASTING,
            version="1.0.0",
            status=ModelStatus.UNTRAINED
        )
        super().__init__(metadata)
        
        self.config = config
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.train_data = None
        self.date_index = None
        self.frequency = None
        
    def _prepare_data(self, data: Union[pd.Series, pd.DataFrame, np.ndarray], date_column: Optional[str] = None) -> pd.Series:
        """Prepare time series data."""
        # Convert numpy array to Series first
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = pd.Series(data)
            elif data.ndim == 2 and data.shape[1] == 1:
                data = pd.Series(data.flatten())
            else:
                # Take first column if multidimensional
                data = pd.Series(data[:, 0])
        
        if isinstance(data, pd.DataFrame):
            if date_column:
                data = data.set_index(date_column)
            # Assume first column is the target if multiple columns
            if len(data.columns) > 1:
                data = data.iloc[:, 0]
            else:
                squeezed = data.squeeze()
                if isinstance(squeezed, pd.Series):
                    data = squeezed
                else:
                    # If squeezing results in a scalar, convert it to a Series
                    data = pd.Series([squeezed] if not isinstance(squeezed, (list, np.ndarray)) else squeezed)
        
        # Ensure we have a datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if hasattr(data.index, 'to_datetime'):
                data.index = pd.to_datetime(data.index)
            else:
                # Create a simple datetime index
                data.index = pd.date_range(start='2020-01-01', periods=len(data), freq='D')
        
        # Infer frequency
        self.frequency = pd.infer_freq(data.index)
        if self.frequency is None:
            self.frequency = 'D'  # Default to daily
        
        # Store for later use
        self.date_index = data.index
        
        return data
    
    def _create_model(self, data: pd.Series):
        """Create the appropriate time series model."""
        config = self.config
        
        if config.model_type == TimeSeriesModel.ARIMA:
            if not STATSMODELS_AVAILABLE:
                raise ImportError("statsmodels required for ARIMA models")
            from statsmodels.tsa.arima.model import ARIMA
            return ARIMA(data, order=config.order)
        
        elif config.model_type == TimeSeriesModel.SARIMA:
            if not STATSMODELS_AVAILABLE:
                raise ImportError("statsmodels required for SARIMA models")
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            return SARIMAX(data, order=config.order, seasonal_order=config.seasonal_order)
        
        elif config.model_type == TimeSeriesModel.EXPONENTIAL_SMOOTHING:
            if not STATSMODELS_AVAILABLE:
                raise ImportError("statsmodels required for Exponential Smoothing")
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            return ExponentialSmoothing(
                data,
                trend=config.trend,
                seasonal=config.seasonal,
                seasonal_periods=config.seasonal_periods,
                damped_trend=config.damped_trend
            )
        
        elif config.model_type == TimeSeriesModel.PROPHET:
            if not PROPHET_AVAILABLE:
                raise ImportError("prophet required for Prophet models")
            from prophet import Prophet
            model = Prophet(
                seasonality_mode=config.seasonality_mode,
                changepoint_prior_scale=config.changepoint_prior_scale,
                seasonality_prior_scale=config.seasonality_prior_scale
            )
            if config.yearly_seasonality:
                model.add_seasonality(name='yearly', period=365.25, fourier_order=10)
            if config.weekly_seasonality:
                model.add_seasonality(name='weekly', period=7, fourier_order=3)
            if config.daily_seasonality:
                model.add_seasonality(name='daily', period=1, fourier_order=3)
            return model
        
        elif config.model_type == TimeSeriesModel.LSTM:
            if not TENSORFLOW_AVAILABLE:
                raise ImportError("tensorflow required for LSTM models")
            return self._create_lstm_model()
        
        else:
            raise ValueError(f"Unsupported time series model: {config.model_type}")
    
    def _create_lstm_model(self):
        """Create LSTM model for time series forecasting."""
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Dropout
        from keras.optimizers import Adam
        config = self.config
        
        model = Sequential([
            LSTM(config.lstm_units, return_sequences=True, input_shape=(config.lookback_window, 1)),
            Dropout(config.dropout_rate),
            LSTM(config.lstm_units, return_sequences=False),
            Dropout(config.dropout_rate),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def _prepare_lstm_data(self, data: pd.Series):
        """Prepare data for LSTM training."""
        # Scale the data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(np.asarray(data).reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        lookback = self.config.lookback_window
        
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'TimeSeriesAnalyzer':
        """Fit the time series model."""
        try:
            self.metadata.status = ModelStatus.TRAINING

            # Prepare data
            ts_data = self._prepare_data(X)
            self.train_data = ts_data

            # Create and fit model
            if self.config.model_type == TimeSeriesModel.LSTM:
                # Special handling for LSTM
                X_lstm, y_lstm = self._prepare_lstm_data(ts_data)
                X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))

                self.model = self._create_lstm_model()
                self.model.fit(X_lstm, y_lstm, epochs=self.config.epochs,
                              batch_size=self.config.batch_size, verbose='0')

            elif self.config.model_type == TimeSeriesModel.PROPHET:
                # Special handling for Prophet
                prophet_data = pd.DataFrame({
                    'ds': ts_data.index,
                    'y': ts_data.values
                })
                self.model = self._create_model(ts_data)
                self.model.fit(prophet_data)

            else:
                # Statistical models
                self.model = self._create_model(ts_data)
                self.model = self.model.fit(ts_data)

            self.is_fitted = True
            self.metadata.status = ModelStatus.FAILED if not self.is_fitted else ModelStatus.TRAINED
            return self

        except Exception as e:
            self.metadata.status = ModelStatus.FAILED
            raise e
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Generate predictions using the fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # For time series, prediction is essentially forecasting
        # Use the length of X as the number of steps to forecast
        if isinstance(X, pd.DataFrame):
            steps = len(X)
        elif isinstance(X, pd.Series):
            steps = len(X)
        else:
            steps = 1
            
        forecast_result = self.forecast(steps=steps, return_conf_int=False)
        return forecast_result.forecast
    
    def forecast(self, steps: int = 10, return_conf_int: bool = True) -> ForecastResult:
        """Generate forecasts."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        config = self.config
        
        if config.model_type == TimeSeriesModel.LSTM:
            return self._forecast_lstm(steps)
        elif config.model_type == TimeSeriesModel.PROPHET:
            return self._forecast_prophet(steps, return_conf_int)
        else:
            return self._forecast_statistical(steps, return_conf_int)
    
    def _forecast_statistical(self, steps: int, return_conf_int: bool) -> ForecastResult:
        """Forecast using statistical models."""
        # Use get_forecast for ARIMA/SARIMAX/ExponentialSmoothing fitted models
        get_forecast_method = getattr(self.model, "get_forecast", None)
        if get_forecast_method is not None:
            forecast_obj = get_forecast_method(steps=steps)
            forecast = forecast_obj.predicted_mean.values
            conf_int = forecast_obj.conf_int().values if return_conf_int else None
        else:
            # Fallback for models without get_forecast (should not happen for supported models)
            raise AttributeError("Model does not support forecasting via get_forecast")
        
        # Ensure self.date_index and self.frequency are set
        if self.date_index is None or len(self.date_index) == 0:
            raise ValueError("date_index is not set or empty")
        last_date = self.date_index[-1]
        freq = self.frequency if self.frequency is not None else 'D'
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=steps,
            freq=freq
        )
        
        return ForecastResult(
            forecast=forecast,
            confidence_intervals=conf_int,
            model_fit=self.model,
            forecast_dates=forecast_dates
        )
    
    def _forecast_prophet(self, steps: int, return_conf_int: bool) -> ForecastResult:
        """Forecast using Prophet."""
        # Ensure self.model is a Prophet instance
        from prophet import Prophet
        if not isinstance(self.model, Prophet):
            raise TypeError("self.model is not a Prophet instance. Check model creation logic.")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=steps)
        forecast = self.model.predict(future)
        
        # Extract forecast values and confidence intervals as numpy arrays
        forecast_values = np.asarray(forecast['yhat'].iloc[-steps:].values)
        conf_int = None
        if return_conf_int:
            conf_int = np.asarray(forecast[['yhat_lower', 'yhat_upper']].iloc[-steps:].values)
        
        # Create forecast dates
        forecast_dates = pd.DatetimeIndex(forecast['ds'].iloc[-steps:].values)
        
        return ForecastResult(
            forecast=forecast_values,
            confidence_intervals=conf_int,
            model_fit=self.model,
            forecast_dates=forecast_dates
        )
    
    def _forecast_lstm(self, steps: int) -> ForecastResult:
        """Forecast using LSTM."""
        # Check required attributes
        if self.scaler is None:
            raise ValueError("Scaler is not initialized. Fit the model first.")
        if self.train_data is None:
            raise ValueError("Training data is not available. Fit the model first.")
        if self.model is None:
            raise ValueError("Model is not initialized. Fit the model first.")
        if self.date_index is None or len(self.date_index) == 0:
            raise ValueError("date_index is not set or empty")
        if self.frequency is None:
            self.frequency = 'D'

        lookback = self.config.lookback_window
        # Ensure train_data is a Series and get last lookback values
        train_tail = self.train_data[-lookback:] if hasattr(self.train_data, '__getitem__') else pd.Series(self.train_data).tail(lookback)
        # Convert to numpy array and reshape
        last_sequence = np.asarray(train_tail).reshape(-1, 1)
        last_sequence = self.scaler.transform(last_sequence)

        forecasts = []
        current_sequence = last_sequence.copy()

        for _ in range(steps):
            # Reshape for prediction
            X = current_sequence.reshape((1, lookback, 1))
            # Predict next value using safe attribute access
            predict_method = getattr(self.model, "predict", None)
            if predict_method is None:
                raise AttributeError("Model does not have predict method")
            pred = predict_method(X)
            # If pred is a tuple, get the first element
            if isinstance(pred, tuple):
                pred_value = pred[0][0]
            else:
                pred_value = pred[0][0]
            forecasts.append(pred_value)
            # Update sequence
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = pred_value

        # Inverse transform forecasts
        forecasts_arr = np.array(forecasts).reshape(-1, 1)
        forecasts_inv = self.scaler.inverse_transform(forecasts_arr).flatten()

        # Create forecast dates
        last_date = self.date_index[-1]
        freq = self.frequency if self.frequency is not None else 'D'
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=steps,
            freq=freq
        )

        return ForecastResult(
            forecast=forecasts_inv,
            forecast_dates=forecast_dates
        )
    
    def evaluate(self, test_data: Union[pd.Series, pd.DataFrame], 
                 date_column: Optional[str] = None) -> Dict[str, float]:
        """Evaluate model performance on test data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Prepare test data
        # Only pass date_column if it is a string
        if isinstance(date_column, str):
            test_ts = self._prepare_data(test_data, date_column)
        else:
            test_ts = self._prepare_data(test_data)
        
        # Generate forecasts for test period
        forecast_result = self.forecast(steps=len(test_ts), return_conf_int=False)
        
        # Convert to numpy float arrays for metrics
        y_true = np.asarray(test_ts.values, dtype=float)
        y_pred = np.asarray(forecast_result.forecast, dtype=float)
        
        # Calculate metrics
        mse = float(mean_squared_error(y_true, y_pred))
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        
        # MAPE (Mean Absolute Percentage Error)
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
        
        # R² score
        r2 = float(r2_score(y_true, y_pred))
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2
        }
    
    def plot_forecast(self, forecast_result: ForecastResult, figsize: Tuple[int, int] = (12, 6)):
        """Plot time series with forecast."""
        plt.figure(figsize=figsize)

        # Plot historical data if available
        if self.date_index is not None and self.train_data is not None:
            plt.plot(
                np.asarray(self.date_index),
                np.asarray(self.train_data.values),
                label='Historical', color='blue', linewidth=2
            )

        # Plot forecast if available
        if forecast_result.forecast_dates is not None and forecast_result.forecast is not None:
            plt.plot(
                np.asarray(forecast_result.forecast_dates),
                np.asarray(forecast_result.forecast),
                label='Forecast', color='red', linewidth=2, linestyle='--'
            )

        # Plot confidence intervals if available
        if (
            forecast_result.confidence_intervals is not None
            and forecast_result.forecast_dates is not None
            and forecast_result.confidence_intervals.shape[1] == 2
        ):
            plt.fill_between(
                np.asarray(forecast_result.forecast_dates),
                forecast_result.confidence_intervals[:, 0],
                forecast_result.confidence_intervals[:, 1],
                alpha=0.3, color='red', label='Confidence Interval'
            )

        plt.title(f'{self.config.model_type.value.title()} Forecast')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def decompose_series(self, model: str = 'additive', period: Optional[int] = None) -> Dict[str, pd.Series]:
        """Decompose time series into trend, seasonal, and residual components."""
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for time series decomposition")
        
        # Ensure period is an integer
        if period is None or not isinstance(period, int):
            period = int(getattr(self.config, "seasonal_periods", 12))
        
        # Import seasonal_decompose locally to avoid unbound error
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        decomposition = seasonal_decompose(
            self.train_data, 
            model=model, 
            period=period
        )
        
        # Replace None with empty Series to match return type
        def safe_series(s):
            if isinstance(s, pd.Series) and s is not None:
                return s
            # Use train_data index if available, otherwise create default
            default_index = self.train_data.index if self.train_data is not None else pd.DatetimeIndex([])
            return pd.Series([], dtype=float, index=default_index)
        
        return {
            'original': safe_series(self.train_data),
            'trend': safe_series(decomposition.trend),
            'seasonal': safe_series(decomposition.seasonal),
            'residual': safe_series(decomposition.resid)
        }
    
    def plot_decomposition(self, model: str = 'additive', period: Optional[int] = None, 
                          figsize: Tuple[int, int] = (12, 10)):
        """Plot time series decomposition."""
        components = self.decompose_series(model, period)
        
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        
        # Original series
        axes[0].plot(components['original'])
        axes[0].set_title('Original Time Series')
        axes[0].grid(True, alpha=0.3)
        
        # Trend
        axes[1].plot(components['trend'])
        axes[1].set_title('Trend Component')
        axes[1].grid(True, alpha=0.3)
        
        # Seasonal
        axes[2].plot(components['seasonal'])
        axes[2].set_title('Seasonal Component')
        axes[2].grid(True, alpha=0.3)
        
        # Residual
        axes[3].plot(components['residual'])
        axes[3].set_title('Residual Component')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return components


class AutoTimeSeriesAnalyzer:
    """Automatic time series model selection and hyperparameter tuning."""
    
    def __init__(self, models_to_try: Optional[List[TimeSeriesModel]] = None):
        """Initialize auto analyzer."""
        if models_to_try is None:
            models_to_try = [
                TimeSeriesModel.ARIMA,
                TimeSeriesModel.EXPONENTIAL_SMOOTHING
            ]
            if PROPHET_AVAILABLE:
                models_to_try.append(TimeSeriesModel.PROPHET)
        
        self.models_to_try = models_to_try
        self.best_model = None
        self.best_score = float('inf')
        self.results = []
    
    def fit(self, data: Union[pd.Series, pd.DataFrame], 
            validation_split: float = 0.2, 
            date_column: Optional[str] = None) -> 'AutoTimeSeriesAnalyzer':
        """Automatically fit and select best model."""
        
        # Prepare data
        if isinstance(data, pd.DataFrame):
            if date_column:
                data = data.set_index(date_column)
            data = data.iloc[:, 0]  # First column
        
        # Split data
        split_idx = int(len(data) * (1 - validation_split))
        train_data = data.iloc[:split_idx]
        val_data = data.iloc[split_idx:]
        
        print(f"🔍 Auto Time Series Analysis - Testing {len(self.models_to_try)} models")
        print(f"📊 Train: {len(train_data)} samples, Validation: {len(val_data)} samples")
        
        for model_type in self.models_to_try:
            try:
                print(f"\n🔬 Testing {model_type.value.title()}...")
                
                # Create model with default config
                config = TimeSeriesConfig(model_type=model_type)
                
                # Adjust config based on model type
                if model_type == TimeSeriesModel.EXPONENTIAL_SMOOTHING:
                    # Try different seasonality settings
                    for seasonal in [None, 'add', 'mul']:
                        try:
                            config.seasonal = seasonal
                            analyzer = TimeSeriesAnalyzer(config)
                            # Convert to proper types - DataFrame for X, Series for y
                            train_X = pd.DataFrame(train_data) if not isinstance(train_data, pd.DataFrame) else train_data
                            train_y = train_data if isinstance(train_data, pd.Series) else train_data.iloc[:, 0]
                            analyzer.fit(train_X, train_y)
                            
                            # Evaluate
                            metrics = analyzer.evaluate(val_data)
                            
                            result = {
                                'model_type': model_type,
                                'config': config,
                                'analyzer': analyzer,
                                'metrics': metrics,
                                'rmse': metrics['rmse']
                            }
                            
                            self.results.append(result)
                            
                            print(f"   ✅ {model_type.value.title()} (seasonal={seasonal}): "
                                  f"RMSE = {metrics['rmse']:.4f}, MAPE = {metrics['mape']:.2f}%")
                            
                            if metrics['rmse'] < self.best_score:
                                self.best_score = metrics['rmse']
                                self.best_model = result
                                
                        except Exception as e:
                            print(f"   ❌ {model_type.value.title()} (seasonal={seasonal}) failed: {e}")
                
                else:
                    # Single configuration for other models
                    analyzer = TimeSeriesAnalyzer(config)
                    # Convert to proper types - DataFrame for X, Series for y
                    train_X = pd.DataFrame(train_data) if not isinstance(train_data, pd.DataFrame) else train_data
                    train_y = train_data if isinstance(train_data, pd.Series) else train_data.iloc[:, 0]
                    analyzer.fit(train_X, train_y)
                    
                    # Evaluate
                    metrics = analyzer.evaluate(val_data)
                    
                    result = {
                        'model_type': model_type,
                        'config': config,
                        'analyzer': analyzer,
                        'metrics': metrics,
                        'rmse': metrics['rmse']
                    }
                    
                    self.results.append(result)
                    
                    print(f"   ✅ {model_type.value.title()}: "
                          f"RMSE = {metrics['rmse']:.4f}, MAPE = {metrics['mape']:.2f}%")
                    
                    if metrics['rmse'] < self.best_score:
                        self.best_score = metrics['rmse']
                        self.best_model = result
                
            except Exception as e:
                print(f"   ❌ {model_type.value.title()} failed: {e}")
        
        # Sort results by RMSE
        self.results.sort(key=lambda x: x['rmse'])
        
        if self.best_model is not None:
            print(f"\n🏆 Best Model: {self.best_model['model_type'].value.title()}")
            print(f"📈 Best RMSE: {self.best_score:.4f}")
        else:
            print("\n🏆 No model was successfully fitted.")
        
        return self
    
    def get_best_model(self) -> TimeSeriesAnalyzer:
        """Get the best performing model."""
        if self.best_model is None:
            raise ValueError("No models have been fitted yet")
        return self.best_model['analyzer']
    
    def get_results_summary(self) -> pd.DataFrame:
        """Get summary of all model results."""
        if not self.results:
            raise ValueError("No models have been fitted yet")
        
        summary_data = []
        for result in self.results:
            summary_data.append({
                'Model': result['model_type'].value.title(),
                'RMSE': result['rmse'],
                'MAE': result['metrics']['mae'],
                'MAPE': result['metrics']['mape'],
                'R²': result['metrics']['r2']
            })
        
        return pd.DataFrame(summary_data)


# Utility functions
def generate_sample_timeseries(
    n_points: int = 365,
    trend: float = 0.1,
    seasonal_amplitude: float = 10,
    noise_level: float = 1,
    start_date: str = '2020-01-01',
    freq: str = 'D'
) -> pd.Series:
    """Generate sample time series data."""
    dates = pd.date_range(start=start_date, periods=n_points, freq=freq)
    
    # Trend component
    trend_component = np.arange(n_points) * trend
    
    # Seasonal component
    seasonal_component = seasonal_amplitude * np.sin(2 * np.pi * np.arange(n_points) / 365.25)
    
    # Noise component
    noise_component = np.random.normal(0, noise_level, n_points)
    
    # Combine components
    values = 100 + trend_component + seasonal_component + noise_component
    
    return pd.Series(values, index=dates, name='value')


def create_time_series_analyzer(
    model_type: TimeSeriesModel = TimeSeriesModel.ARIMA,
    **kwargs
) -> TimeSeriesAnalyzer:
    """Create a time series analyzer with specified configuration."""
    config = TimeSeriesConfig(model_type=model_type, **kwargs)
    return TimeSeriesAnalyzer(config)


def detect_seasonality(data: pd.Series, max_period: int = 50) -> Dict[str, Any]:
    """Detect seasonality in time series data."""
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels required for seasonality detection")
    
    # Import acf locally to ensure it is bound
    from statsmodels.tsa.stattools import acf

    # Calculate autocorrelation
    autocorr = acf(data.dropna(), nlags=max_period, fft=True)
    
    # Find peaks (potential seasonal periods)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(autocorr[1:], height=0.1)  # Skip lag 0
    peaks += 1  # Adjust for skipped lag 0
    
    # Get the most significant peak
    if len(peaks) > 0:
        peak_values = autocorr[peaks]
        strongest_peak_idx = np.argmax(peak_values)
        strongest_period = peaks[strongest_peak_idx]
        strength = peak_values[strongest_peak_idx]
    else:
        strongest_period = None
        strength = 0
    
    return {
        'seasonal_period': strongest_period,
        'strength': strength,
        'all_peaks': peaks,
        'autocorr': autocorr
    }


def test_stationarity(data: pd.Series) -> Dict[str, Any]:
    """Test stationarity using Augmented Dickey-Fuller test."""
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels required for stationarity test")
    
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(data.dropna())
    
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[2],
        'is_stationary': result[1] < 0.05
    }


# Export all classes and functions
__all__ = [
    # Main classes
    'TimeSeriesAnalyzer', 'AutoTimeSeriesAnalyzer',
    
    # Configuration classes
    'TimeSeriesConfig', 'ForecastResult',
    
    # Enums
    'TimeSeriesModel', 'SeasonalityType', 'TrendType',
    
    # Factory functions
    'create_time_series_analyzer',
    
    # Utility functions
    'generate_sample_timeseries', 'detect_seasonality', 'test_stationarity'
]
