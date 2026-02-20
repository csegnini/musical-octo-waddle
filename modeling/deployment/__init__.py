"""
Model Deployment and Serving Package

This package provides comprehensive model deployment capabilities including:
- Flask API creation and management
- Model serving and inference endpoints
- REST API generation for ML models
- Production deployment utilities
- Model monitoring and logging
- Batch prediction services
- Real-time inference servers
"""

import os
import sys
import json
import pickle
import joblib
import logging
import datetime
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd

# Flask and web serving imports
try:
    from flask import Flask, request, jsonify, render_template_string
    from flask_cors import CORS
    import threading
    import time
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("⚠️ Flask not available - some deployment features disabled")
    # Provide stubs for Flask symbols to avoid NameError
    class Flask:
        def __init__(self, *args, **kwargs):
            raise ImportError("Flask is not installed")
    def request():
        raise ImportError("Flask is not installed")
    def jsonify(*args, **kwargs):
        return args[0] if args else {}
    def render_template_string(*args, **kwargs):
        return "Flask not installed"
    class CORS:
        def __init__(self, *args, **kwargs):
            pass
    class threading:
        class Thread:
            def __init__(self, *args, **kwargs):
                pass
            def start(self):
                pass
            @property
            def daemon(self):
                return False
            @daemon.setter
            def daemon(self, value):
                pass
    def time():
        import time as _time
        return _time.time()

# Model serialization
try:
    import cloudpickle
    CLOUDPICKLE_AVAILABLE = True
except ImportError:
    CLOUDPICKLE_AVAILABLE = False

# Add base module to path
base_path = os.path.join(os.path.dirname(__file__), '..')
if base_path not in sys.path:
    sys.path.insert(0, base_path)

try:
    from .base import ModelMetadata, ModelStatus, ProblemType, ModelType
except ImportError:
    # If base.py does not exist, define stubs for compatibility
    class ModelMetadata: pass
    class ModelStatus: pass
    class ProblemType: pass
    class ModelType: pass


class SerializationFormat(Enum):
    """Model serialization formats."""
    PICKLE = "pickle"
    JOBLIB = "joblib"
    CLOUDPICKLE = "cloudpickle"
    JSON = "json"


class DeploymentMode(Enum):
    """Deployment modes."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class APIMethod(Enum):
    """API endpoint methods."""
    POST = "POST"
    GET = "GET"
    PUT = "PUT"
    DELETE = "DELETE"


@dataclass
class ModelEndpoint:
    """Configuration for a model endpoint."""
    name: str
    model_path: str
    endpoint_path: str
    methods: List[APIMethod] = field(default_factory=lambda: [APIMethod.POST])
    description: str = ""
    input_schema: Optional[Dict] = None
    output_schema: Optional[Dict] = None
    preprocessing_func: Optional[Callable] = None
    postprocessing_func: Optional[Callable] = None
    rate_limit: Optional[int] = None
    authentication_required: bool = False


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    name: str
    version: str = "1.0.0"
    host: str = "localhost"
    port: int = 5000
    debug: bool = False
    mode: DeploymentMode = DeploymentMode.DEVELOPMENT
    cors_enabled: bool = True
    logging_level: str = "INFO"
    max_request_size: int = 16 * 1024 * 1024  # 16MB
    request_timeout: int = 30
    model_cache_size: int = 10
    enable_monitoring: bool = True
    health_check_interval: int = 60


class ModelSerializer:
    """Handles model serialization and deserialization."""
    
    def __init__(self, format: SerializationFormat = SerializationFormat.JOBLIB):
        self.format = format
        
    def serialize(self, model: Any, filepath: str) -> bool:
        """Serialize a model to file."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            if self.format == SerializationFormat.PICKLE:
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)
            elif self.format == SerializationFormat.JOBLIB:
                joblib.dump(model, filepath)
            elif self.format == SerializationFormat.CLOUDPICKLE:
                if not CLOUDPICKLE_AVAILABLE:
                    raise ImportError("cloudpickle not available")
                with open(filepath, 'wb') as f:
                    cloudpickle.dump(model, f)
            else:
                raise ValueError(f"Unsupported format: {self.format}")
                
            return True
        except Exception as e:
            logging.error(f"Serialization failed: {e}")
            return False
    
    def deserialize(self, filepath: str) -> Any:
        """Deserialize a model from file."""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Model file not found: {filepath}")
                
            if self.format == SerializationFormat.PICKLE:
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            elif self.format == SerializationFormat.JOBLIB:
                return joblib.load(filepath)
            elif self.format == SerializationFormat.CLOUDPICKLE:
                if not CLOUDPICKLE_AVAILABLE:
                    raise ImportError("cloudpickle not available")
                with open(filepath, 'rb') as f:
                    return cloudpickle.load(f)
            else:
                raise ValueError(f"Unsupported format: {self.format}")
                
        except Exception as e:
            logging.error(f"Deserialization failed: {e}")
            raise


class ModelCache:
    """In-memory model cache for fast inference."""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
        self.serializer = ModelSerializer()
        
    def get(self, model_path: str) -> Any:
        """Get model from cache or load from disk."""
        if model_path in self.cache:
            self.access_times[model_path] = datetime.datetime.now()
            return self.cache[model_path]
        
        # Load model and add to cache
        model = self.serializer.deserialize(model_path)
        self._add_to_cache(model_path, model)
        return model
    
    def _add_to_cache(self, model_path: str, model: Any):
        """Add model to cache with LRU eviction."""
        if len(self.cache) >= self.max_size:
            # Remove least recently used model
            lru_path = min(self.access_times.keys(), 
                          key=lambda k: self.access_times[k])
            del self.cache[lru_path]
            del self.access_times[lru_path]
        
        self.cache[model_path] = model
        self.access_times[model_path] = datetime.datetime.now()
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.access_times.clear()
    
    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)


class RequestLogger:
    """Logs API requests and responses."""
    
    def __init__(self, log_file: str = "deployment.log"):
        self.log_file = log_file
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def log_request(self, endpoint: str, method: str, data: Dict, 
                   response: Dict, duration: float, status_code: int):
        """Log API request details."""
        log_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'endpoint': endpoint,
            'method': method,
            'input_size': len(str(data)),
            'output_size': len(str(response)),
            'duration_ms': duration * 1000,
            'status_code': status_code
        }
        logging.info(f"API_REQUEST: {json.dumps(log_entry)}")


class FlaskModelServer:
    """Flask-based model serving server."""
    
    def __init__(self, config: DeploymentConfig):
        if not FLASK_AVAILABLE:
            raise ImportError("Flask not available for model serving")
            
        self.config = config
        self.app = Flask(__name__)
        self.model_cache = ModelCache(config.model_cache_size)
        self.logger = RequestLogger()
        self.endpoints = {}
        self.is_running = False
        
        # Configure Flask app
        if config.cors_enabled:
            CORS(self.app)
        
        # Setup default routes
        self._setup_default_routes()
        
    def add_model_endpoint(self, endpoint: ModelEndpoint):
        """Add a model endpoint to the server."""
        self.endpoints[endpoint.name] = endpoint
        
        # Create Flask route
        route_path = f"/models/{endpoint.endpoint_path}"
        methods = [method.value for method in endpoint.methods]
        
        def endpoint_handler():
            return self._handle_model_request(endpoint)
        
        endpoint_handler.__name__ = f"endpoint_{endpoint.name}"
        self.app.add_url_rule(
            route_path, 
            endpoint_handler.__name__, 
            endpoint_handler, 
            methods=methods
        )
        
    def _setup_default_routes(self):
        """Setup default API routes."""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.datetime.now().isoformat(),
                'version': self.config.version,
                'cache_size': self.model_cache.size(),
                'endpoints': list(self.endpoints.keys())
            })
        
        @self.app.route('/info', methods=['GET'])
        def server_info():
            """Server information endpoint."""
            return jsonify({
                'name': self.config.name,
                'version': self.config.version,
                'mode': self.config.mode.value,
                'endpoints': {
                    name: {
                        'path': f"/models/{ep.endpoint_path}",
                        'methods': [m.value for m in ep.methods],
                        'description': ep.description
                    }
                    for name, ep in self.endpoints.items()
                }
            })
        
        @self.app.route('/models', methods=['GET'])
        def list_models():
            """List available model endpoints."""
            return jsonify({
                'models': [
                    {
                        'name': ep.name,
                        'endpoint': f"/models/{ep.endpoint_path}",
                        'description': ep.description,
                        'methods': [m.value for m in ep.methods]
                    }
                    for ep in self.endpoints.values()
                ]
            })
        
        @self.app.route('/', methods=['GET'])
        def home():
            """Home page with API documentation."""
            html_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>{{ config.name }} - Model API Server</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .header { color: #333; border-bottom: 2px solid #007acc; padding-bottom: 10px; }
                    .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
                    .method { background: #007acc; color: white; padding: 3px 8px; border-radius: 3px; font-size: 12px; }
                    code { background: #e8e8e8; padding: 2px 4px; border-radius: 3px; }
                </style>
            </head>
            <body>
                <h1 class="header">🚀 {{ config.name }} - Model API Server</h1>
                <p><strong>Version:</strong> {{ config.version }}</p>
                <p><strong>Mode:</strong> {{ config.mode.value }}</p>
                <p><strong>Status:</strong> 🟢 Running</p>
                
                <h2>📋 Available Endpoints</h2>
                
                <div class="endpoint">
                    <h3>🏥 Health Check</h3>
                    <p><span class="method">GET</span> <code>/health</code></p>
                    <p>Check server health and status</p>
                </div>
                
                <div class="endpoint">
                    <h3>ℹ️ Server Information</h3>
                    <p><span class="method">GET</span> <code>/info</code></p>
                    <p>Get detailed server information and endpoints</p>
                </div>
                
                <div class="endpoint">
                    <h3>📝 List Models</h3>
                    <p><span class="method">GET</span> <code>/models</code></p>
                    <p>List all available model endpoints</p>
                </div>
                
                {% for endpoint in endpoints.values() %}
                <div class="endpoint">
                    <h3>🤖 {{ endpoint.name }}</h3>
                    <p>
                        {% for method in endpoint.methods %}
                        <span class="method">{{ method.value }}</span>
                        {% endfor %}
                        <code>/models/{{ endpoint.endpoint_path }}</code>
                    </p>
                    <p>{{ endpoint.description or "Model inference endpoint" }}</p>
                </div>
                {% endfor %}
                
                <h2>📊 Usage Example</h2>
                <pre><code>
import requests

# Health check
response = requests.get('http://{{ config.host }}:{{ config.port }}/health')
print(response.json())

# Model prediction (example)
data = {'features': [1, 2, 3, 4, 5]}
response = requests.post('http://{{ config.host }}:{{ config.port }}/models/predict', json=data)
print(response.json())
                </code></pre>
            </body>
            </html>
            """
            return render_template_string(html_template, 
                                        config=self.config, 
                                        endpoints=self.endpoints)
    
    def _handle_model_request(self, endpoint: ModelEndpoint):
        """Handle model inference request."""
        start_time = time.time()
        
        try:
            # Get request data
            if request.method == 'GET':
                data = request.args.to_dict()
            else:
                data = request.get_json()
            
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            # Load model from cache
            model = self.model_cache.get(endpoint.model_path)
            
            # Preprocess data if function provided
            if endpoint.preprocessing_func:
                data = endpoint.preprocessing_func(data)
            
            # Make prediction
            if hasattr(model, 'predict'):
                if isinstance(data.get('features'), list):
                    features = np.array(data['features']).reshape(1, -1)
                    prediction = model.predict(features)
                elif 'X' in data:
                    prediction = model.predict(data['X'])
                else:
                    return jsonify({'error': 'Invalid input format'}), 400
            else:
                return jsonify({'error': 'Model does not support prediction'}), 500
            
            # Convert prediction to JSON-serializable format
            if isinstance(prediction, np.ndarray):
                prediction = prediction.tolist()
            
            # Postprocess if function provided
            result = {'prediction': prediction}
            if endpoint.postprocessing_func:
                result = endpoint.postprocessing_func(result)
            
            # Add metadata
            result.update({
                'model_name': endpoint.name,
                'timestamp': datetime.datetime.now().isoformat(),
                'processing_time_ms': (time.time() - start_time) * 1000
            })
            
            # Log request
            self.logger.log_request(
                endpoint.endpoint_path, 
                request.method, 
                data, 
                result, 
                time.time() - start_time, 
                200
            )
            
            return jsonify(result)
            
        except Exception as e:
            error_response = {
                'error': str(e),
                'timestamp': datetime.datetime.now().isoformat(),
                'processing_time_ms': (time.time() - start_time) * 1000
            }
            
            # Log error
            self.logger.log_request(
                endpoint.endpoint_path, 
                request.method, 
                data if 'data' in locals() else {}, 
                error_response, 
                time.time() - start_time, 
                500
            )
            
            return jsonify(error_response), 500
    
    def run(self, threaded: bool = True):
        """Start the Flask server."""
        if self.is_running:
            print("Server is already running")
            return
        
        self.is_running = True
        print(f"🚀 Starting {self.config.name} server...")
        print(f"📍 Server URL: http://{self.config.host}:{self.config.port}")
        print(f"🔧 Mode: {self.config.mode.value}")
        print(f"📊 Endpoints: {len(self.endpoints)} model(s)")
        
        if threaded:
            # Run in separate thread for non-blocking operation
            server_thread = threading.Thread(
                target=self.app.run,
                kwargs={
                    'host': self.config.host,
                    'port': self.config.port,
                    'debug': self.config.debug,
                    'threaded': True
                }
            )
            server_thread.daemon = True
            server_thread.start()
            return server_thread
        else:
            self.app.run(
                host=self.config.host,
                port=self.config.port,
                debug=self.config.debug,
                threaded=True
            )
    
    def stop(self):
        """Stop the server."""
        self.is_running = False
        print("🛑 Server stopped")


class BatchPredictor:
    """Handles batch predictions for large datasets."""
    
    def __init__(self, model_path: str, batch_size: int = 1000):
        self.model_path = model_path
        self.batch_size = batch_size
        self.serializer = ModelSerializer()
        
    def predict_batch(self, X: Union[np.ndarray, pd.DataFrame], 
                     output_file: Optional[str] = None) -> np.ndarray:
        """Make predictions on large dataset in batches."""
        model = self.serializer.deserialize(self.model_path)
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        n_samples = X.shape[0]
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        predictions = []
        
        print(f"🔄 Processing {n_samples} samples in {n_batches} batches...")
        
        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, n_samples)
            
            batch_X = X[start_idx:end_idx]
            batch_pred = model.predict(batch_X)
            predictions.append(batch_pred)
            
            if (i + 1) % 10 == 0:
                print(f"   Processed batch {i + 1}/{n_batches}")
        
        result = np.concatenate(predictions)
        
        if output_file:
            np.save(output_file, result)
            print(f"💾 Predictions saved to {output_file}")
        
        return result


def create_flask_server(name: str, 
                       host: str = "localhost", 
                       port: int = 5000,
                       mode: DeploymentMode = DeploymentMode.DEVELOPMENT) -> FlaskModelServer:
    """Create a Flask model server with default configuration."""
    config = DeploymentConfig(
        name=name,
        host=host,
        port=port,
        mode=mode
    )
    return FlaskModelServer(config)


def deploy_model(model: Any, 
                model_name: str,
                endpoint_path: str = "predict",
                model_dir: str = "models",
                server_config: Optional[DeploymentConfig] = None) -> FlaskModelServer:
    """Deploy a trained model as a Flask API endpoint."""
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Serialize model
    model_path = os.path.join(model_dir, f"{model_name}.joblib")
    serializer = ModelSerializer()
    
    if not serializer.serialize(model, model_path):
        raise RuntimeError(f"Failed to serialize model {model_name}")
    
    # Create server configuration
    if server_config is None:
        server_config = DeploymentConfig(name=f"{model_name}_api")
    
    # Create server
    server = FlaskModelServer(server_config)
    
    # Create endpoint
    endpoint = ModelEndpoint(
        name=model_name,
        model_path=model_path,
        endpoint_path=endpoint_path,
        description=f"Inference endpoint for {model_name} model"
    )
    
    server.add_model_endpoint(endpoint)
    
    print(f"✅ Model {model_name} deployed successfully!")
    print(f"📍 Endpoint: /models/{endpoint_path}")
    print(f"💾 Model saved to: {model_path}")
    
    return server


def create_model_endpoint(name: str, 
                         model_path: str, 
                         endpoint_path: str,
                         description: str = "",
                         preprocessing_func: Optional[Callable] = None,
                         postprocessing_func: Optional[Callable] = None) -> ModelEndpoint:
    """Create a model endpoint configuration."""
    return ModelEndpoint(
        name=name,
        model_path=model_path,
        endpoint_path=endpoint_path,
        description=description,
        preprocessing_func=preprocessing_func,
        postprocessing_func=postprocessing_func
    )


def save_model(model: Any, 
               filepath: str, 
               format: SerializationFormat = SerializationFormat.JOBLIB,
               metadata: Optional[Dict] = None) -> bool:
    """Save a model to disk with optional metadata."""
    serializer = ModelSerializer(format)
    
    # Save model
    success = serializer.serialize(model, filepath)
    
    # Save metadata if provided
    if success and metadata:
        metadata_path = filepath.replace('.joblib', '_metadata.json').replace('.pkl', '_metadata.json')
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save metadata: {e}")
    
    return success


def load_model(filepath: str, 
               format: SerializationFormat = SerializationFormat.JOBLIB) -> Any:
    """Load a model from disk."""
    serializer = ModelSerializer(format)
    return serializer.deserialize(filepath)


# Convenience functions for common deployment patterns
def quick_deploy(model: Any, 
                model_name: str,
                port: int = 5000,
                start_server: bool = True) -> FlaskModelServer:
    """Quickly deploy a model with minimal configuration."""
    server = deploy_model(
        model=model,
        model_name=model_name,
        server_config=DeploymentConfig(
            name=f"{model_name}_quick_api",
            port=port,
            debug=True
        )
    )
    
    if start_server:
        server.run(threaded=True)
        print(f"🌐 Server running at: http://localhost:{port}")
        print(f"📋 API docs available at: http://localhost:{port}")
        print(f"🏥 Health check: http://localhost:{port}/health")
        print(f"🤖 Model endpoint: http://localhost:{port}/models/predict")
    
    return server


# Export main classes and functions
__all__ = [
    'FlaskModelServer',
    'ModelEndpoint', 
    'DeploymentConfig',
    'ModelSerializer',
    'ModelCache',
    'BatchPredictor',
    'SerializationFormat',
    'DeploymentMode',
    'APIMethod',
    'create_flask_server',
    'deploy_model',
    'create_model_endpoint',
    'save_model',
    'load_model',
    'quick_deploy'
]
