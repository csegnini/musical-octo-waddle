"""
Workflow Orchestration Utilities

This module provides utility classes and functions for building, validating,
and serializing workflows with additional helper functionality.
"""

import json
import yaml
import pickle
from typing import Dict, List, Any, Optional, Callable, Union, Set
from pathlib import Path
from datetime import datetime
import logging
import importlib
import inspect
from dataclasses import asdict

from .core import Task, Workflow, TaskResult, WorkflowResult, TaskStatus, WorkflowStatus

logger = logging.getLogger(__name__)


class WorkflowBuilder:
    """Builder pattern for creating workflows with fluent interface."""
    
    def __init__(self, name: str, description: str = ""):
        self.workflow = Workflow(name=name, description=description)
        self.current_tasks: List[Task] = []
        self.logger = logging.getLogger(__name__)
    
    def add_task(
        self, 
        name: str, 
        function: Callable, 
        dependencies: Optional[List[str]] = None,
        **kwargs
    ) -> 'WorkflowBuilder':
        """Add a task to the workflow."""
        task = Task(name=name, function=function, dependencies=dependencies or [], **kwargs)
        self.workflow.add_task(task)
        self.current_tasks = [task]
        return self
    
    def chain(self, *tasks: Task) -> 'WorkflowBuilder':
        """Chain tasks in sequence (each depends on the previous)."""
        if not tasks:
            return self
        
        previous_task = None
        for task in tasks:
            if previous_task:
                if previous_task.task_id not in task.dependencies:
                    task.dependencies.append(previous_task.task_id)
            self.workflow.add_task(task)
            previous_task = task
        
        self.current_tasks = list(tasks)
        return self
    
    def parallel(self, *tasks: Task) -> 'WorkflowBuilder':
        """Add tasks to run in parallel."""
        for task in tasks:
            self.workflow.add_task(task)
        self.current_tasks = list(tasks)
        return self
    
    def depends_on(self, *task_ids: str) -> 'WorkflowBuilder':
        """Add dependencies to the current tasks."""
        for task in self.current_tasks:
            for task_id in task_ids:
                if task_id not in task.dependencies:
                    task.dependencies.append(task_id)
        return self
    
    def with_config(self, **config) -> 'WorkflowBuilder':
        """Add configuration to the workflow."""
        self.workflow.config.update(config)
        return self
    
    def with_metadata(self, **metadata) -> 'WorkflowBuilder':
        """Add metadata to the workflow."""
        self.workflow.metadata.update(metadata)
        return self
    
    def with_retry_policy(self, **retry_config) -> 'WorkflowBuilder':
        """Add retry policy to current tasks."""
        for task in self.current_tasks:
            task.retry_policy = retry_config
        return self
    
    def with_timeout(self, timeout: float) -> 'WorkflowBuilder':
        """Add timeout to current tasks."""
        for task in self.current_tasks:
            task.timeout = timeout
        return self
    
    def with_resources(self, **resources) -> 'WorkflowBuilder':
        """Add resource requirements to current tasks."""
        for task in self.current_tasks:
            task.resources.update(resources)
        return self
    
    def build(self) -> Workflow:
        """Build and return the workflow."""
        return self.workflow


class TaskFactory:
    """Factory for creating common types of tasks."""
    
    @staticmethod
    def create_python_function_task(
        name: str,
        function: Callable,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Task:
        """Create a task from a Python function."""
        return Task(
            name=name,
            function=function,
            parameters=parameters or {},
            **kwargs
        )
    
    @staticmethod
    def create_shell_command_task(
        name: str,
        command: str,
        shell: bool = True,
        capture_output: bool = True,
        **kwargs
    ) -> Task:
        """Create a task that executes a shell command."""
        import subprocess
        
        def shell_executor(**params):
            # Substitute parameters in command
            formatted_command = command
            for key, value in params.items():
                formatted_command = formatted_command.replace(f"{{{key}}}", str(value))
            
            result = subprocess.run(
                formatted_command,
                shell=shell,
                capture_output=capture_output,
                text=True
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Command failed: {result.stderr}")
            
            return result.stdout
        
        return Task(
            name=name,
            function=shell_executor,
            metadata={'command': command, 'type': 'shell'},
            **kwargs
        )
    
    @staticmethod
    def create_http_request_task(
        name: str,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Task:
        """Create a task that makes HTTP requests."""
        import requests
        
        def http_executor(**params):
            # Prepare request parameters
            request_kwargs = {
                'method': method.upper(),
                'url': url,
                'headers': headers or {}
            }
            
            # Add data/json from parameters
            if 'data' in params:
                request_kwargs['data'] = params['data']
            if 'json' in params:
                request_kwargs['json'] = params['json']
            
            response = requests.request(**request_kwargs)
            response.raise_for_status()
            
            try:
                return response.json()
            except ValueError:
                return response.text
        
        return Task(
            name=name,
            function=http_executor,
            metadata={'url': url, 'method': method, 'type': 'http'},
            **kwargs
        )
    
    @staticmethod
    def create_file_operation_task(
        name: str,
        operation: str,  # read, write, copy, move, delete
        file_path: str,
        **kwargs
    ) -> Task:
        """Create a task for file operations."""
        import shutil
        
        def file_executor(**params):
            path = Path(file_path)
            
            if operation == "read":
                if path.suffix.lower() == '.json':
                    with open(path, 'r') as f:
                        return json.load(f)
                elif path.suffix.lower() in ['.yml', '.yaml']:
                    with open(path, 'r') as f:
                        return yaml.safe_load(f)
                else:
                    with open(path, 'r') as f:
                        return f.read()
            
            elif operation == "write":
                data = params.get('data')
                if path.suffix.lower() == '.json':
                    with open(path, 'w') as f:
                        json.dump(data, f, indent=2)
                elif path.suffix.lower() in ['.yml', '.yaml']:
                    with open(path, 'w') as f:
                        yaml.dump(data, f)
                else:
                    with open(path, 'w') as f:
                        f.write(str(data))
                return str(path)
            
            elif operation == "copy":
                dest = params.get('destination')
                if dest is None:
                    raise ValueError("Destination path must be provided for copy operation")
                shutil.copy2(path, str(dest))
                return str(dest)
            
            elif operation == "move":
                dest = params.get('destination')
                if dest is None:
                    raise ValueError("Destination path must be provided for move operation")
                shutil.move(str(path), str(dest))
                return str(dest)
            
            elif operation == "delete":
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
                return str(path)
            
            else:
                raise ValueError(f"Unknown file operation: {operation}")
        
        return Task(
            name=name,
            function=file_executor,
            metadata={'file_path': file_path, 'operation': operation, 'type': 'file'},
            **kwargs
        )
    
    @staticmethod
    def create_data_processing_task(
        name: str,
        processor_func: Callable,
        input_key: str = "data",
        **kwargs
    ) -> Task:
        """Create a task for data processing operations."""
        def data_executor(**params):
            input_data = params.get(input_key)
            if input_data is None:
                raise ValueError(f"No input data found with key: {input_key}")
            
            return processor_func(input_data)
        
        return Task(
            name=name,
            function=data_executor,
            metadata={'type': 'data_processing', 'input_key': input_key},
            **kwargs
        )


class WorkflowValidator:
    """Validates workflow definitions and configurations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_workflow(self, workflow: Workflow) -> List[str]:
        """Comprehensive workflow validation."""
        issues = []
        
        # Basic validation
        issues.extend(workflow.validate())
        
        # Additional validations
        issues.extend(self._validate_task_functions(workflow))
        issues.extend(self._validate_resource_consistency(workflow))
        issues.extend(self._validate_timeout_settings(workflow))
        issues.extend(self._validate_retry_policies(workflow))
        
        return issues
    
    def _validate_task_functions(self, workflow: Workflow) -> List[str]:
        """Validate that all task functions are callable."""
        issues = []
        
        for task in workflow.tasks.values():
            if not callable(task.function):
                issues.append(f"Task {task.name} has non-callable function")
            else:
                # Check function signature
                try:
                    sig = inspect.signature(task.function)
                    # Basic signature validation could be added here
                except Exception as e:
                    issues.append(f"Task {task.name} function signature error: {e}")
        
        return issues
    
    def _validate_resource_consistency(self, workflow: Workflow) -> List[str]:
        """Validate resource requirements are consistent."""
        issues = []
        
        total_cpu = 0
        total_memory = 0
        
        for task in workflow.tasks.values():
            if task.resources:
                cpu = task.resources.get('cpu_cores', 0)
                memory = task.resources.get('memory_mb', 0)
                
                if cpu < 0:
                    issues.append(f"Task {task.name} has negative CPU requirement")
                if memory < 0:
                    issues.append(f"Task {task.name} has negative memory requirement")
                
                total_cpu += cpu
                total_memory += memory
        
        # Check for reasonable resource limits
        if total_cpu > 100:  # Assuming reasonable limit
            issues.append(f"Total CPU requirements ({total_cpu}) seem excessive")
        if total_memory > 100000:  # 100GB limit
            issues.append(f"Total memory requirements ({total_memory}MB) seem excessive")
        
        return issues
    
    def _validate_timeout_settings(self, workflow: Workflow) -> List[str]:
        """Validate timeout settings."""
        issues = []
        
        for task in workflow.tasks.values():
            if task.timeout is not None:
                if task.timeout <= 0:
                    issues.append(f"Task {task.name} has invalid timeout: {task.timeout}")
                elif task.timeout > 86400:  # 24 hours
                    issues.append(f"Task {task.name} has very long timeout: {task.timeout} seconds")
        
        return issues
    
    def _validate_retry_policies(self, workflow: Workflow) -> List[str]:
        """Validate retry policy configurations."""
        issues = []
        
        for task in workflow.tasks.values():
            if task.retry_policy:
                policy = task.retry_policy
                
                max_attempts = policy.get('max_attempts', 1)
                if max_attempts < 1:
                    issues.append(f"Task {task.name} has invalid max_attempts: {max_attempts}")
                elif max_attempts > 10:
                    issues.append(f"Task {task.name} has excessive max_attempts: {max_attempts}")
                
                base_delay = policy.get('base_delay', 1.0)
                if base_delay < 0:
                    issues.append(f"Task {task.name} has negative base_delay: {base_delay}")
        
        return issues


class WorkflowSerializer:
    """Serializes and deserializes workflows to/from various formats."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.function_registry: Dict[str, Callable] = {}
    
    def register_function(self, name: str, function: Callable):
        """Register a function for serialization/deserialization."""
        self.function_registry[name] = function
    
    def serialize_workflow(self, workflow: Workflow, format: str = "json") -> str:
        """Serialize workflow to string."""
        # Convert workflow to dictionary
        workflow_dict = self._workflow_to_dict(workflow)
        
        if format.lower() == "json":
            return json.dumps(workflow_dict, indent=2, default=str)
        elif format.lower() == "yaml":
            return yaml.dump(workflow_dict, default_flow_style=False)
        elif format.lower() == "pickle":
            import base64
            return base64.b64encode(pickle.dumps(workflow_dict)).decode()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def deserialize_workflow(self, data: str, format: str = "json") -> Workflow:
        """Deserialize workflow from string."""
        if format.lower() == "json":
            workflow_dict = json.loads(data)
        elif format.lower() == "yaml":
            workflow_dict = yaml.safe_load(data)
        elif format.lower() == "pickle":
            import base64
            workflow_dict = pickle.loads(base64.b64decode(data))
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return self._dict_to_workflow(workflow_dict)
    
    def save_workflow(self, workflow: Workflow, file_path: str):
        """Save workflow to file."""
        path = Path(file_path)
        format = path.suffix.lower().lstrip('.')
        
        serialized = self.serialize_workflow(workflow, format)
        
        with open(path, 'w') as f:
            f.write(serialized)
        
        self.logger.info(f"Saved workflow to {file_path}")
    
    def load_workflow(self, file_path: str) -> Workflow:
        """Load workflow from file."""
        path = Path(file_path)
        format = path.suffix.lower().lstrip('.')
        
        with open(path, 'r') as f:
            data = f.read()
        
        workflow = self.deserialize_workflow(data, format)
        self.logger.info(f"Loaded workflow from {file_path}")
        return workflow
    
    def _workflow_to_dict(self, workflow: Workflow) -> Dict[str, Any]:
        """Convert workflow to dictionary representation."""
        workflow_dict = {
            'workflow_id': workflow.workflow_id,
            'name': workflow.name,
            'description': workflow.description,
            'config': workflow.config,
            'metadata': workflow.metadata,
            'created_at': workflow.created_at.isoformat(),
            'tasks': {}
        }
        
        for task_id, task in workflow.tasks.items():
            task_dict = {
                'task_id': task.task_id,
                'name': task.name,
                'function_name': self._get_function_name(task.function),
                'dependencies': task.dependencies,
                'parameters': task.parameters,
                'timeout': task.timeout,
                'retry_policy': task.retry_policy,
                'resources': task.resources,
                'metadata': task.metadata,
                'created_at': task.created_at.isoformat()
            }
            workflow_dict['tasks'][task_id] = task_dict
        
        return workflow_dict
    
    def _dict_to_workflow(self, workflow_dict: Dict[str, Any]) -> Workflow:
        """Convert dictionary to workflow object."""
        workflow = Workflow(
            name=workflow_dict['name'],
            workflow_id=workflow_dict['workflow_id'],
            description=workflow_dict['description'],
            config=workflow_dict['config'],
            metadata=workflow_dict['metadata']
        )
        
        # Restore creation time
        if 'created_at' in workflow_dict:
            workflow.created_at = datetime.fromisoformat(workflow_dict['created_at'])
        
        # Restore tasks
        for task_id, task_dict in workflow_dict['tasks'].items():
            function = self._get_function_by_name(task_dict['function_name'])
            
            task = Task(
                name=task_dict['name'],
                function=function,
                task_id=task_dict['task_id'],
                dependencies=task_dict['dependencies'],
                parameters=task_dict['parameters'],
                timeout=task_dict['timeout'],
                retry_policy=task_dict['retry_policy'],
                resources=task_dict['resources'],
                metadata=task_dict['metadata']
            )
            
            # Restore creation time
            if 'created_at' in task_dict:
                task.created_at = datetime.fromisoformat(task_dict['created_at'])
            
            workflow.add_task(task)
        
        return workflow
    
    def _get_function_name(self, function: Callable) -> str:
        """Get a serializable name for a function."""
        # Check if function is in registry
        for name, registered_func in self.function_registry.items():
            if registered_func == function:
                return name
        
        # Try to get module and function name
        if hasattr(function, '__module__') and hasattr(function, '__name__'):
            return f"{function.__module__}.{function.__name__}"
        
        # Fallback to string representation
        return str(function)
    
    def _get_function_by_name(self, function_name: str) -> Callable:
        """Get function object by name."""
        # Check registry first
        if function_name in self.function_registry:
            return self.function_registry[function_name]
        
        # Try to import from module
        if '.' in function_name:
            module_name, func_name = function_name.rsplit('.', 1)
            try:
                module = importlib.import_module(module_name)
                return getattr(module, func_name)
            except (ImportError, AttributeError) as e:
                self.logger.warning(f"Could not import function {function_name}: {e}")
        
        # Return placeholder function
        def placeholder_function(**kwargs):
            raise NotImplementedError(f"Function {function_name} not available")
        
        return placeholder_function


# Utility functions for common workflow patterns
def create_linear_workflow(name: str, task_functions: List[Callable], task_names: Optional[List[str]] = None) -> Workflow:
    """Create a linear workflow where tasks execute in sequence."""
    builder = WorkflowBuilder(name, "Linear workflow with sequential task execution")
    
    if task_names is None:
        task_names = [f"task_{i+1}" for i in range(len(task_functions))]
    
    if len(task_names) != len(task_functions):
        raise ValueError("Number of task names must match number of functions")
    
    tasks = []
    for i, (task_name, func) in enumerate(zip(task_names, task_functions)):
        task = Task(name=task_name, function=func)
        tasks.append(task)
    
    return builder.chain(*tasks).build()


def create_parallel_workflow(name: str, task_functions: List[Callable], task_names: Optional[List[str]] = None) -> Workflow:
    """Create a workflow where all tasks execute in parallel."""
    builder = WorkflowBuilder(name, "Parallel workflow with concurrent task execution")
    
    if task_names is None:
        task_names = [f"task_{i+1}" for i in range(len(task_functions))]
    
    if len(task_names) != len(task_functions):
        raise ValueError("Number of task names must match number of functions")
    
    tasks = []
    for task_name, func in zip(task_names, task_functions):
        task = Task(name=task_name, function=func)
        tasks.append(task)
    
    return builder.parallel(*tasks).build()


def create_map_reduce_workflow(
    name: str,
    map_function: Callable,
    reduce_function: Callable,
    data_chunks: List[Any],
    chunk_names: Optional[List[str]] = None
) -> Workflow:
    """Create a map-reduce style workflow."""
    builder = WorkflowBuilder(name, "Map-Reduce workflow pattern")
    
    if chunk_names is None:
        chunk_names = [f"chunk_{i+1}" for i in range(len(data_chunks))]
    
    # Create map tasks
    map_tasks = []
    for i, (chunk_name, chunk_data) in enumerate(zip(chunk_names, data_chunks)):
        map_task = Task(
            name=f"map_{chunk_name}",
            function=map_function,
            parameters={'data': chunk_data}
        )
        map_tasks.append(map_task)
        builder.workflow.add_task(map_task)
    
    # Create reduce task that depends on all map tasks
    reduce_task = Task(
        name="reduce",
        function=reduce_function,
        dependencies=[task.task_id for task in map_tasks]
    )
    builder.workflow.add_task(reduce_task)
    
    return builder.build()
