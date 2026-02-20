"""
Modeling Package

This package provides comprehensive machine learning modeling# Import time series models
from .time_series import (
    TimeSeriesAnalyzer, AutoTimeSeriesAnalyzer,
    TimeSeriesConfig, ForecastResult,
    TimeSeriesModel, SeasonalityType, TrendType,
    create_time_series_analyzer,
    generate_sample_timeseries, detect_seasonality, test_stationarity
)ities
for the Scientist system. It includes:

- Base classes and interfaces for all models
- Regression models (linear, polynomial, regularized)
- Evaluation and validation tools
- Model management and utilities

The package is designed to be modular and extensible, allowing for easy
addition of new model types and evaluation methods.
"""

from .base import (
    BaseModel,
    BasePreprocessor, 
    BaseEvaluator,
    ModelMetadata,
    TrainingConfig,
    DataInfo,
    ModelType,
    ModelStatus,
    ProblemType
)

from .regression import (
    LinearRegressionModel,
    PolynomialRegressionModel,
    RegressionModelFactory,
    create_linear_regression,
#     create_ridge_regression,
#     create_lasso_regression,
#     create_elastic_net,
#     create_polynomial_regression
 )

# Import classification models
from .classification import (
    LogisticRegressionModel,
    SVMModel,
    DecisionTreeModel,
    RandomForestModel,
    ClassificationModelFactory,
    create_logistic_regression,
    create_svm,
    create_decision_tree,
    create_random_forest
)

# Import ensemble methods
from .ensemble import (
    VotingEnsembleModel,
    BaggingEnsembleModel,
    BoostingEnsembleModel,
    StackingEnsembleModel,
    create_voting_ensemble,
    create_bagging_ensemble,
    create_boosting_ensemble,
    create_stacking_ensemble,
    create_random_forest as create_random_forest_ensemble
)

# Import neural networks
from .neural_networks import (
    FeedforwardNetwork,
    ConvolutionalNetwork,
    create_feedforward_network,
    create_cnn,
    create_mlp_classifier,
    create_mlp_regressor,
    ActivationType,
    OptimizerType,
    NetworkArchitecture
)

# Import unsupervised learning
from .unsupervised import (
    ClusteringModel, DimensionalityReductionModel, AnomalyDetectionModel,
    ClusteringConfig, DimensionalityReductionConfig, AnomalyDetectionConfig,
    ClusteringAlgorithm, DimensionalityReductionAlgorithm, AnomalyDetectionAlgorithm,
    create_clustering_model, create_dimensionality_reduction_model, create_anomaly_detection_model,
    find_optimal_clusters, plot_cluster_optimization
)

# Import time series analysis
from .time_series import (
    TimeSeriesAnalyzer, AutoTimeSeriesAnalyzer,
    TimeSeriesConfig, ForecastResult,
    TimeSeriesModel, SeasonalityType, TrendType,
    create_time_series_analyzer,
    generate_sample_timeseries, detect_seasonality, test_stationarity
)

# Import statistical analysis
from .statistical_analysis import (
    HypothesisTestAnalyzer,
    RegressionAnalyzer,
    StatisticalTestResult,
    RegressionResult,
    TestType,
    RegressionType,
    CorrelationType,
    create_hypothesis_tester,
    create_regression_analyzer,
    quick_t_test,
    quick_correlation,
    quick_regression
)

# Import deployment and serving
from .deployment import (
    FlaskModelServer,
    ModelEndpoint, 
    DeploymentConfig,
    ModelSerializer,
    ModelCache,
    BatchPredictor,
    SerializationFormat,
    DeploymentMode,
    APIMethod,
    create_flask_server,
    deploy_model,
    create_model_endpoint,
    save_model,
    load_model,
    quick_deploy
)

# Import AutoML and hyperparameter optimization
from .automl import (
    AutoMLPipeline,
    HyperparameterOptimizer,
    AutoMLConfig,
    OptimizationResult,
    OptimizationAlgorithm,
    AutoMLMode,
    FeatureSelectionMethod,
    auto_classify,
    auto_regress,
    optimize_hyperparameters
)

# Import computer vision
from .computer_vision import (
    ImageProcessor,
    ImageAugmentor, 
    FeatureExtractor,
    ImageSegmentor,
    CNNClassifier,
    TransferLearningClassifier,
    ImageInfo,
    DetectionResult,
    SegmentationResult,
    ImageFormat,
    ColorSpace,
    FeatureType,
    AugmentationType,
    create_cnn_classifier,
    create_transfer_learning_classifier,
    generate_sample_images
)

from .evaluation import (
    RegressionEvaluator,
    ClassificationEvaluator,
    ModelValidator,
    EvaluationResult,
    MetricType
)

__all__ = [
    # Base classes
    'BaseModel',
    'BasePreprocessor',
    'BaseEvaluator',
    
    # Data structures
    'ModelMetadata',
    'TrainingConfig',
    'DataInfo',
    'EvaluationResult',
    
    # Enums
    'ModelType',
    'ModelStatus', 
    'ProblemType',
    'MetricType',
    
    # Regression models
    'LinearRegressionModel',
    'PolynomialRegressionModel',
    'RegressionModelFactory',
    
    # Classification models
    'LogisticRegressionModel',
    'SVMModel', 
    'DecisionTreeModel',
    'RandomForestModel',
    'ClassificationModelFactory',
    
    # Ensemble models
    'VotingEnsembleModel',
    'BaggingEnsembleModel',
    'BoostingEnsembleModel',
    'StackingEnsembleModel',
    
    # Neural network models
    'FeedforwardNetwork',
    'ConvolutionalNetwork',
    
    # Unsupervised learning models
    'ClusteringModel',
    'DimensionalityReductionModel', 
    'AnomalyDetectionModel',
    
    # Time series models
    'TimeSeriesAnalyzer',
    'AutoTimeSeriesAnalyzer',
    
    # Statistical analysis
    'HypothesisTestAnalyzer',
    'RegressionAnalyzer',
    'StatisticalTestResult',
    'RegressionResult',
    'TestType',
    'RegressionType',
    'CorrelationType',
    
    # Deployment and serving
    'FlaskModelServer',
    'ModelEndpoint', 
    'DeploymentConfig',
    'ModelSerializer',
    'ModelCache',
    'BatchPredictor',
    'SerializationFormat',
    'DeploymentMode',
    'APIMethod',
    
    # AutoML and optimization
    'AutoMLPipeline',
    'HyperparameterOptimizer',
    'AutoMLConfig',
    'OptimizationResult',
    'OptimizationAlgorithm',
    'AutoMLMode',
    'FeatureSelectionMethod',
    
    # Computer vision
    'ImageProcessor',
    'ImageAugmentor', 
    'FeatureExtractor',
    'ImageSegmentor',
    'CNNClassifier',
    'TransferLearningClassifier',
    'ImageInfo',
    'DetectionResult',
    'SegmentationResult',
    'ImageFormat',
    'ColorSpace',
    'FeatureType',
    'AugmentationType',
    
    # Evaluation tools
    'RegressionEvaluator',
    'ClassificationEvaluator',
    'ModelValidator',
    
    # Convenience functions
    'create_linear_regression',
    'create_polynomial_regression',
    'create_ridge_regression', 
    'create_lasso_regression',
    'create_elastic_net',
    'create_logistic_regression',
    'create_svm',
    'create_decision_tree',
    'create_random_forest',
    'create_voting_ensemble',
    'create_bagging_ensemble',
    'create_boosting_ensemble',
    'create_stacking_ensemble',
    'create_random_forest_ensemble',
    'create_feedforward_network',
    'create_cnn',
    'create_mlp_classifier',
    'create_mlp_regressor',
    'create_clustering_model',
    'create_dimensionality_reduction_model',
    'create_anomaly_detection_model',
    'create_time_series_analyzer',
    'find_optimal_clusters',
    'plot_cluster_optimization',
    'generate_sample_timeseries',
    'detect_seasonality',
    'test_stationarity',
    'create_hypothesis_tester',
    'create_regression_analyzer',
    'quick_t_test',
    'quick_correlation',
    'quick_regression',
    'create_flask_server',
    'deploy_model',
    'create_model_endpoint',
    'save_model',
    'load_model',
    'quick_deploy',
    'auto_classify',
    'auto_regress',
    'optimize_hyperparameters',
    'evaluate_model'
]

__version__ = '1.0.0'
__author__ = 'Multi-Agent Scientist System'

# Package-level convenience functions

def create_linear_regression(**kwargs):
    """Create a linear regression model."""
    return RegressionModelFactory.create_linear_regression(**kwargs)

def create_polynomial_regression(degree=2, **kwargs):
    """Create a polynomial regression model."""
    return RegressionModelFactory.create_polynomial_regression(degree=degree, **kwargs)

def create_ridge_regression(alpha=1.0, **kwargs):
    """Create a ridge regression model."""
    return RegressionModelFactory.create_ridge_regression(alpha=alpha, **kwargs)

def create_lasso_regression(alpha=1.0, **kwargs):
    """Create a lasso regression model.""" 
    return RegressionModelFactory.create_lasso_regression(alpha=alpha, **kwargs)

def create_elastic_net(alpha=1.0, l1_ratio=0.5, **kwargs):
    """Create an elastic net regression model."""
    return RegressionModelFactory.create_elastic_net(alpha=alpha, l1_ratio=l1_ratio, **kwargs)

def evaluate_model(model, X, y, model_type=ModelType.REGRESSION):
    """Evaluate a model comprehensively."""
    if model_type == ModelType.REGRESSION:
        evaluator = RegressionEvaluator()
        predictions = model.predict(X)
        return evaluator.evaluate(y, predictions)
    else:
        evaluator = ClassificationEvaluator()
        predictions = model.predict(X)
        return evaluator.evaluate(y, predictions)
