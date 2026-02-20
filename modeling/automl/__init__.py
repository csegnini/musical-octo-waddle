"""
AutoML and Hyperparameter Optimization Package

This package provides comprehensive automated machine learning capabilities including:
- Automated model selection and comparison
- Hyperparameter optimization using various algorithms
- Feature engineering and selection automation
- Pipeline optimization and automation
- Cross-validation and model evaluation
- Advanced optimization algorithms (Grid Search, Random Search, Bayesian, Genetic)
- Multi-objective optimization
- Early stopping and pruning
"""

import os
import sys
import json
import time
import warnings
import itertools
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score, 
    StratifiedKFold, KFold, train_test_split
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.pipeline import Pipeline

# Optimization libraries
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

# Machine learning models
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Add base module to path
base_path = os.path.join(os.path.dirname(__file__), '..')
if base_path not in sys.path:
    sys.path.insert(0, base_path)

# Define stubs for compatibility (not used in this module)
class ModelMetadata: pass
class ModelStatus: pass
class ProblemType: pass
class ModelType: pass


class OptimizationAlgorithm(Enum):
    """Hyperparameter optimization algorithms."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    OPTUNA = "optuna"
    HYPEROPT = "hyperopt"
    GENETIC = "genetic"
    SUCCESSIVE_HALVING = "successive_halving"


class AutoMLMode(Enum):
    """AutoML operation modes."""
    FAST = "fast"           # Quick evaluation, fewer models
    BALANCED = "balanced"   # Balance between speed and thoroughness
    THOROUGH = "thorough"   # Comprehensive search, more time
    CUSTOM = "custom"       # User-defined configuration


class FeatureSelectionMethod(Enum):
    """Feature selection methods."""
    SELECT_K_BEST = "select_k_best"
    RECURSIVE_ELIMINATION = "rfe"
    LASSO_SELECTION = "lasso"
    TREE_IMPORTANCE = "tree_importance"
    MUTUAL_INFO = "mutual_info"


@dataclass
class OptimizationResult:
    """Results from hyperparameter optimization."""
    best_params: Dict[str, Any]
    best_score: float
    best_model: Any
    optimization_history: List[Dict] = field(default_factory=list)
    total_trials: int = 0
    optimization_time: float = 0.0
    algorithm_used: str = ""
    cross_val_scores: Optional[List[float]] = None
    feature_importance: Optional[Dict[str, float]] = None


@dataclass
class AutoMLConfig:
    """Configuration for AutoML pipeline."""
    mode: AutoMLMode = AutoMLMode.BALANCED
    optimization_algorithm: OptimizationAlgorithm = OptimizationAlgorithm.RANDOM_SEARCH
    cv_folds: int = 5
    scoring_metric: str = "accuracy"  # or "neg_mean_squared_error" for regression
    max_trials: int = 100
    timeout_minutes: Optional[int] = 30
    feature_selection: bool = True
    feature_selection_method: FeatureSelectionMethod = FeatureSelectionMethod.SELECT_K_BEST
    max_features: Optional[int] = None
    preprocessing: bool = True
    ensemble_models: bool = True
    neural_networks: bool = True
    early_stopping: bool = True
    random_state: int = 42
    n_jobs: int = -1
    verbose: bool = True


class HyperparameterOptimizer:
    """Advanced hyperparameter optimization."""
    
    def __init__(self, 
                 algorithm: OptimizationAlgorithm = OptimizationAlgorithm.RANDOM_SEARCH,
                 cv_folds: int = 5,
                 scoring: str = "accuracy",
                 random_state: int = 42,
                 n_jobs: int = -1):
        self.algorithm = algorithm
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.optimization_history = []
        
    def optimize(self, 
                model: Any,
                param_space: Dict[str, Any],
                X: np.ndarray,
                y: np.ndarray,
                max_trials: int = 100,
                timeout: Optional[int] = None) -> OptimizationResult:
        """Optimize hyperparameters using specified algorithm."""
        
        start_time = time.time()
        
        if self.algorithm == OptimizationAlgorithm.GRID_SEARCH:
            return self._grid_search_optimize(model, param_space, X, y)
        elif self.algorithm == OptimizationAlgorithm.RANDOM_SEARCH:
            return self._random_search_optimize(model, param_space, X, y, max_trials)
        elif self.algorithm == OptimizationAlgorithm.OPTUNA and OPTUNA_AVAILABLE:
            return self._optuna_optimize(model, param_space, X, y, max_trials, timeout)
        elif self.algorithm == OptimizationAlgorithm.BAYESIAN and SKOPT_AVAILABLE:
            return self._bayesian_optimize(model, param_space, X, y, max_trials)
        elif self.algorithm == OptimizationAlgorithm.HYPEROPT and HYPEROPT_AVAILABLE:
            # Hyperopt optimization not implemented; fallback to random search
            print(f"⚠️ Hyperopt optimization not implemented, using random search")
            return self._random_search_optimize(model, param_space, X, y, max_trials)
        else:
            # Fallback to random search
            print(f"⚠️ {self.algorithm.value} not available, using random search")
            return self._random_search_optimize(model, param_space, X, y, max_trials)
    
    def _grid_search_optimize(self, model, param_space, X, y) -> OptimizationResult:
        """Grid search optimization."""
        grid_search = GridSearchCV(
            model, param_space, cv=self.cv_folds, 
            scoring=self.scoring, n_jobs=self.n_jobs
        )
        
        grid_search.fit(X, y)
        
        return OptimizationResult(
            best_params=grid_search.best_params_,
            best_score=grid_search.best_score_,
            best_model=grid_search.best_estimator_,
            total_trials=len(grid_search.cv_results_['params']),
            algorithm_used="Grid Search",
            cross_val_scores=grid_search.cv_results_['mean_test_score'].tolist()
        )
    
    def _random_search_optimize(self, model, param_space, X, y, max_trials) -> OptimizationResult:
        """Random search optimization."""
        random_search = RandomizedSearchCV(
            model, param_space, n_iter=max_trials, cv=self.cv_folds,
            scoring=self.scoring, n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        
        random_search.fit(X, y)
        
        return OptimizationResult(
            best_params=random_search.best_params_,
            best_score=random_search.best_score_,
            best_model=random_search.best_estimator_,
            total_trials=len(random_search.cv_results_['params']),
            algorithm_used="Random Search",
            cross_val_scores=random_search.cv_results_['mean_test_score'].tolist()
        )
    
    def _optuna_optimize(self, model, param_space, X, y, max_trials, timeout) -> OptimizationResult:
        """Optuna-based optimization."""
        import optuna
        study = optuna.create_study(direction='maximize' if 'accuracy' in self.scoring else 'minimize')
        
        def objective(trial):
            # Convert param space to Optuna suggestions
            params = {}
            for key, value in param_space.items():
                if isinstance(value, list):
                    if all(isinstance(v, int) for v in value):
                        params[key] = trial.suggest_int(key, min(value), max(value))
                    elif all(isinstance(v, float) for v in value):
                        params[key] = trial.suggest_float(key, min(value), max(value))
                    else:
                        params[key] = trial.suggest_categorical(key, value)
                elif isinstance(value, tuple) and len(value) == 2:
                    if isinstance(value[0], int):
                        params[key] = trial.suggest_int(key, value[0], value[1])
                    else:
                        params[key] = trial.suggest_float(key, value[0], value[1])
            
            # Create model with suggested parameters
            model_instance = model.__class__(**params)
            
            # Cross-validation
            if isinstance(self.cv_folds, int):
                if hasattr(y, 'nunique') and len(np.unique(y)) > 1:
                    cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                else:
                    cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            else:
                cv = self.cv_folds
            
            scores = cross_val_score(model_instance, X, y, cv=cv, scoring=self.scoring, n_jobs=self.n_jobs)
            return scores.mean()
        
        study.optimize(objective, n_trials=max_trials, timeout=timeout)
        
        # Rebuild best model
        best_model = model.__class__(**study.best_params)
        best_model.fit(X, y)
        
        return OptimizationResult(
            best_params=study.best_params,
            best_score=study.best_value,
            best_model=best_model,
            total_trials=len(study.trials),
            algorithm_used="Optuna",
            optimization_history=[{
                'trial': i, 
                'score': trial.value, 
                'params': trial.params
            } for i, trial in enumerate(study.trials)]
        )
    
    def _bayesian_optimize(self, model, param_space, X, y, max_trials) -> OptimizationResult:
        """Scikit-optimize Bayesian optimization."""
        # Ensure skopt is available
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize (skopt) is not installed. Please install it to use Bayesian optimization.")

        # Import required skopt components locally to avoid unbound errors
        from skopt import gp_minimize
        from skopt.space import Real, Integer, Categorical
        from skopt.utils import use_named_args

        # Convert param space to skopt format
        dimensions = []
        param_names = []

        for key, value in param_space.items():
            param_names.append(key)
            if isinstance(value, list):
                if all(isinstance(v, (int, float)) for v in value):
                    dimensions.append(Real(min(value), max(value), name=key))
                else:
                    dimensions.append(Categorical(value, name=key))
            elif isinstance(value, tuple) and len(value) == 2:
                if isinstance(value[0], int):
                    dimensions.append(Integer(value[0], value[1], name=key))
                else:
                    dimensions.append(Real(value[0], value[1], name=key))

        @use_named_args(dimensions)
        def objective(**params):
            model_instance = model.__class__(**params)

            if isinstance(self.cv_folds, int):
                if hasattr(y, 'nunique') and len(np.unique(y)) > 1:
                    cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                else:
                    cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            else:
                cv = self.cv_folds

            scores = cross_val_score(model_instance, X, y, cv=cv, scoring=self.scoring, n_jobs=self.n_jobs)
            # Return negative score for minimization
            return -scores.mean() if 'accuracy' in self.scoring else scores.mean()

        result = gp_minimize(objective, dimensions, n_calls=max_trials, random_state=self.random_state)

        if result is not None and hasattr(result, "x") and hasattr(result, "fun") and hasattr(result, "func_vals"):
            # Extract best parameters
            best_params = {param_names[i]: result.x[i] for i in range(len(param_names))}

            # Rebuild best model
            best_model = model.__class__(**best_params)
            best_model.fit(X, y)

            return OptimizationResult(
                best_params=best_params,
                best_score=-result.fun if 'accuracy' in self.scoring else result.fun,
                best_model=best_model,
                total_trials=len(result.func_vals),
                algorithm_used="Bayesian Optimization"
            )
        else:
            # Handle failed optimization gracefully
            warnings.warn("Bayesian optimization failed or returned None. No best parameters found.")
            base_model = model.__class__()
            base_model.fit(X, y)
            scores = cross_val_score(base_model, X, y, cv=self.cv_folds, scoring=self.scoring, n_jobs=self.n_jobs)
            return OptimizationResult(
                best_params={},
                best_score=scores.mean(),
                best_model=base_model,
                total_trials=1,
                algorithm_used="Bayesian Optimization (fallback)"
            )


class AutoMLPipeline:
    """Automated Machine Learning Pipeline."""
    
    def __init__(self, config: Optional[AutoMLConfig] = None):
        self.config = config or AutoMLConfig()
        self.results = {}
        self.best_model = None
        self.feature_selector = None
        self.scaler = None
        
        # Default model configurations
        self.model_configs = self._get_default_model_configs()
        
    def _get_default_model_configs(self) -> Dict[str, Dict]:
        """Get default model configurations for AutoML."""
        configs = {
            'RandomForest': {
                'classifier': RandomForestClassifier,
                'regressor': RandomForestRegressor,
                'param_space': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'GradientBoosting': {
                'classifier': GradientBoostingClassifier,
                'regressor': GradientBoostingRegressor,
                'param_space': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'SVM': {
                'classifier': SVC,
                'regressor': SVR,
                'param_space': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
                }
            },
            'LogisticRegression': {
                'classifier': LogisticRegression,
                'regressor': LinearRegression,
                'param_space': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'solver': ['liblinear', 'lbfgs'],
                    'max_iter': [1000]
                }
            },
            'KNeighbors': {
                'classifier': KNeighborsClassifier,
                'regressor': KNeighborsRegressor,
                'param_space': {
                    'n_neighbors': [3, 5, 7, 11, 15],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree']
                }
            }
        }
        
        if self.config.neural_networks:
            configs['MLP'] = {
                'classifier': MLPClassifier,
                'regressor': MLPRegressor,
                'param_space': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive'],
                    'max_iter': [1000]
                }
            }
        
        # Filter based on mode
        if self.config.mode == AutoMLMode.FAST:
            # Reduce parameter space for faster execution
            for model_name in configs:
                param_space = configs[model_name]['param_space']
                for param, values in param_space.items():
                    if isinstance(values, list) and len(values) > 2:
                        configs[model_name]['param_space'][param] = values[:2]
        
        return configs
    
    def _preprocess_data(self, X: np.ndarray, y: Optional[np.ndarray], fit: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Preprocess data with scaling and feature selection."""
        if self.config.preprocessing:
            if fit:
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = self.scaler.transform(X) if self.scaler else X
        else:
            X_scaled = X

        if self.config.feature_selection and fit and y is not None:
            # Determine problem type
            is_classification = len(np.unique(y)) < 20 and y.dtype in ['int64', 'int32', 'object']

            if self.config.feature_selection_method == FeatureSelectionMethod.SELECT_K_BEST:
                k = self.config.max_features or min(20, X.shape[1])
                score_func = f_classif if is_classification else f_regression
                self.feature_selector = SelectKBest(score_func=score_func, k=k)
                X_selected = self.feature_selector.fit_transform(X_scaled, y)
            else:
                X_selected = X_scaled
        elif self.config.feature_selection and self.feature_selector:
            X_selected = self.feature_selector.transform(X_scaled)
        else:
            X_selected = X_scaled

        # Ensure X_selected is a numpy ndarray (not sparse matrix)
        # Ensure X_selected is a numpy ndarray (not sparse matrix)
        if not isinstance(X_selected, np.ndarray):
            # If it's a scipy sparse matrix, convert to dense
            try:
                import scipy.sparse
                if scipy.sparse.issparse(X_selected):
                    X_selected = X_selected.toarray()
            except ImportError:
                pass
        X_selected = np.asarray(X_selected)
        return X_selected, y
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AutoMLPipeline':
        """Fit the AutoML pipeline."""
        start_time = time.time()
        
        if self.config.verbose:
            print(f"🤖 Starting AutoML Pipeline...")
            print(f"   Mode: {self.config.mode.value}")
            print(f"   Algorithm: {self.config.optimization_algorithm.value}")
            print(f"   Max Trials: {self.config.max_trials}")
            print(f"   CV Folds: {self.config.cv_folds}")
            print(f"   Timeout: {self.config.timeout_minutes} minutes")
        
        # Preprocess data
        X_processed, y_processed = self._preprocess_data(X, y, fit=True)
        
        if self.config.verbose:
            print(f"📊 Data preprocessed: {X_processed.shape[0]} samples, {X_processed.shape[1]} features")
        
        # Determine problem type
        is_classification = len(np.unique(y)) < 20 and y.dtype in ['int64', 'int32', 'object']
        problem_type = 'classifier' if is_classification else 'regressor'
        
        if self.config.verbose:
            print(f"🎯 Detected problem type: {problem_type}")
        
        # Initialize optimizer
        optimizer = HyperparameterOptimizer(
            algorithm=self.config.optimization_algorithm,
            cv_folds=self.config.cv_folds,
            scoring=self.config.scoring_metric,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs
        )
        
        # Evaluate models
        models_to_evaluate = list(self.model_configs.keys())
        if self.config.mode == AutoMLMode.FAST:
            models_to_evaluate = models_to_evaluate[:3]  # Evaluate fewer models
        
        for model_name in models_to_evaluate:
            if self.config.verbose:
                print(f"\n🔬 Evaluating {model_name}...")
            
            try:
                model_config = self.model_configs[model_name]
                model_class = model_config[problem_type]
                param_space = model_config['param_space']
                
                # Create base model
                if model_name == 'LogisticRegression' and problem_type == 'regressor':
                    # Use LinearRegression for regression
                    base_model = LinearRegression()
                    param_space = {}  # LinearRegression has no hyperparameters to tune
                else:
                    base_model = model_class(random_state=self.config.random_state)
                
                # Optimize hyperparameters
                if y_processed is None:
                    raise ValueError("Target variable y cannot be None for model fitting and optimization.")
                if param_space:
                    result = optimizer.optimize(
                        base_model, param_space, X_processed, y_processed,
                        max_trials=self.config.max_trials,
                        timeout=self.config.timeout_minutes * 60 if self.config.timeout_minutes else None
                    )
                else:
                    # No hyperparameters to optimize
                    base_model.fit(X_processed, y_processed)
                    scores = cross_val_score(
                        base_model, X_processed, y_processed, 
                        cv=self.config.cv_folds, scoring=self.config.scoring_metric
                    )
                    result = OptimizationResult(
                        best_params={},
                        best_score=scores.mean(),
                        best_model=base_model,
                        total_trials=1,
                        algorithm_used="No optimization",
                        cross_val_scores=scores.tolist()
                    )
                
                result.optimization_time = time.time() - start_time
                
                # Add feature importance if available
                if hasattr(result.best_model, 'feature_importances_'):
                    if self.feature_selector and hasattr(self.feature_selector, 'get_support'):
                        selected_features = self.feature_selector.get_support(indices=True)
                        result.feature_importance = {
                            f'feature_{i}': importance 
                            for i, importance in zip(selected_features, result.best_model.feature_importances_)
                        }
                    else:
                        result.feature_importance = {
                            f'feature_{i}': importance 
                            for i, importance in enumerate(result.best_model.feature_importances_)
                        }
                
                self.results[model_name] = result
                
                if self.config.verbose:
                    print(f"   ✅ {model_name}: {result.best_score:.4f} score ({result.total_trials} trials)")
                
            except Exception as e:
                if self.config.verbose:
                    print(f"   ❌ {model_name} failed: {e}")
                continue
        
        # Find best model
        if self.results:
            best_model_name = max(self.results.keys(), key=lambda k: self.results[k].best_score)
            self.best_model = self.results[best_model_name].best_model
            
            if self.config.verbose:
                print(f"\n🏆 Best Model: {best_model_name}")
                print(f"   Score: {self.results[best_model_name].best_score:.4f}")
                print(f"   Parameters: {self.results[best_model_name].best_params}")
        
        total_time = time.time() - start_time
        if self.config.verbose:
            print(f"\n⏱️ Total AutoML Time: {total_time:.2f} seconds")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the best model."""
        if self.best_model is None:
            raise ValueError("AutoML pipeline not fitted yet")
        
        X_processed, _ = self._preprocess_data(X, None, fit=False)
        return self.best_model.predict(X_processed)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities (classification only)."""
        if self.best_model is None:
            raise ValueError("AutoML pipeline not fitted yet")
        
        if not hasattr(self.best_model, 'predict_proba'):
            raise ValueError("Best model does not support probability prediction")
        
        X_processed, _ = self._preprocess_data(X, None, fit=False)
        return self.best_model.predict_proba(X_processed)
    
    def get_model_rankings(self) -> List[Tuple[str, float]]:
        """Get models ranked by performance."""
        return sorted(
            [(name, result.best_score) for name, result in self.results.items()],
            key=lambda x: x[1], reverse=True
        )
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from the best model."""
        if self.best_model is None:
            return None
        
        best_model_name = max(self.results.keys(), key=lambda k: self.results[k].best_score)
        return self.results[best_model_name].feature_importance
    
    def generate_report(self) -> str:
        """Generate a comprehensive AutoML report."""
        if not self.results:
            return "No models evaluated yet."
        
        report = ["🤖 AutoML Pipeline Report", "=" * 50]
        
        # Configuration
        report.extend([
            f"\n📋 Configuration:",
            f"   Mode: {self.config.mode.value}",
            f"   Algorithm: {self.config.optimization_algorithm.value}",
            f"   CV Folds: {self.config.cv_folds}",
            f"   Scoring: {self.config.scoring_metric}",
            f"   Max Trials: {self.config.max_trials}"
        ])
        
        # Model Rankings
        rankings = self.get_model_rankings()
        report.extend([
            f"\n🏆 Model Rankings:",
            f"Rank | Model              | Score    | Trials | Time(s)"
        ])
        report.append("-" * 55)
        
        for i, (name, score) in enumerate(rankings, 1):
            result = self.results[name]
            report.append(
                f"{i:4d} | {name:<18} | {score:.4f}   | {result.total_trials:6d} | {result.optimization_time:.2f}"
            )
        
        # Best Model Details
        best_name = rankings[0][0]
        best_result = self.results[best_name]
        report.extend([
            f"\n🥇 Best Model: {best_name}",
            f"   Score: {best_result.best_score:.4f}",
            f"   Parameters: {best_result.best_params}",
            f"   Trials: {best_result.total_trials}",
            f"   Algorithm: {best_result.algorithm_used}"
        ])
        
        # Feature Importance
        feature_importance = self.get_feature_importance()
        if feature_importance:
            report.extend([f"\n🔍 Top Feature Importances:"])
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:10]:
                report.append(f"   {feature}: {importance:.4f}")
        
        return "\n".join(report)


# Convenience functions
def auto_classify(X: np.ndarray, 
                 y: np.ndarray,
                 mode: AutoMLMode = AutoMLMode.BALANCED,
                 max_trials: int = 50) -> AutoMLPipeline:
    """Automated classification pipeline."""
    config = AutoMLConfig(
        mode=mode,
        max_trials=max_trials,
        scoring_metric="accuracy"
    )
    
    pipeline = AutoMLPipeline(config)
    return pipeline.fit(X, y)


def auto_regress(X: np.ndarray, 
                y: np.ndarray,
                mode: AutoMLMode = AutoMLMode.BALANCED,
                max_trials: int = 50) -> AutoMLPipeline:
    """Automated regression pipeline."""
    config = AutoMLConfig(
        mode=mode,
        max_trials=max_trials,
        scoring_metric="neg_mean_squared_error"
    )
    
    pipeline = AutoMLPipeline(config)
    return pipeline.fit(X, y)


def optimize_hyperparameters(model: Any,
                           param_space: Dict[str, Any],
                           X: np.ndarray,
                           y: np.ndarray,
                           algorithm: OptimizationAlgorithm = OptimizationAlgorithm.RANDOM_SEARCH,
                           max_trials: int = 100) -> OptimizationResult:
    """Quick hyperparameter optimization."""
    optimizer = HyperparameterOptimizer(algorithm=algorithm)
    return optimizer.optimize(model, param_space, X, y, max_trials)


# Export main classes and functions
__all__ = [
    'AutoMLPipeline',
    'HyperparameterOptimizer',
    'AutoMLConfig',
    'OptimizationResult',
    'OptimizationAlgorithm',
    'AutoMLMode',
    'FeatureSelectionMethod',
    'auto_classify',
    'auto_regress',
    'optimize_hyperparameters'
]
