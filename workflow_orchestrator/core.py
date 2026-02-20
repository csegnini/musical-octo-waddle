"""
Core Workflow Orchestration Module

This module provides the fundamental classes and interfaces for workflow orchestration
including tasks, workflows, execution results, and the main workflow engine.
"""

import uuid
import time
from typing import Dict, List, Any, Optional, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import threading
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class WorkflowStatus(Enum):
    """Workflow execution status."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class TaskResult:
    """Results from task execution."""
    task_id: str
    status: TaskStatus
    output: Any = None
    error: Optional[Exception] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time: Optional[float] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Calculate task duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def is_successful(self) -> bool:
        """Check if task completed successfully."""
        return self.status == TaskStatus.COMPLETED
    
    @property
    def is_failed(self) -> bool:
        """Check if task failed."""
        return self.status == TaskStatus.FAILED


@dataclass
class WorkflowResult:
    """Results from workflow execution."""
    workflow_id: str
    status: WorkflowStatus
    task_results: Dict[str, TaskResult] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Calculate workflow duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def successful_tasks(self) -> List[str]:
        """Get list of successful task IDs."""
        return [task_id for task_id, result in self.task_results.items() 
                if result.is_successful]
    
    @property
    def failed_tasks(self) -> List[str]:
        """Get list of failed task IDs."""
        return [task_id for task_id, result in self.task_results.items() 
                if result.is_failed]
    
    @property
    def success_rate(self) -> float:
        """Calculate task success rate."""
        if not self.task_results:
            return 0.0
        return len(self.successful_tasks) / len(self.task_results)


class Task:
    """Represents a single task in a workflow."""
    
    def __init__(
        self,
        name: str,
        function: Callable,
        task_id: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        resources: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        condition: Optional[Callable] = None
    ):
        self.task_id = task_id or str(uuid.uuid4())
        self.name = name
        self.function = function
        self.dependencies = dependencies or []
        self.parameters = parameters or {}
        self.timeout = timeout
        self.retry_policy = retry_policy or {}
        self.resources = resources or {}
        self.metadata = metadata or {}
        self.condition = condition  # Conditional execution
        
        # Runtime state
        self.status = TaskStatus.PENDING
        self.result: Optional[TaskResult] = None
        self.created_at = datetime.now()
        
    def can_execute(self, completed_tasks: Set[str]) -> bool:
        """Check if task can be executed based on dependencies."""
        if self.condition and not self.condition():
            return False
        return all(dep in completed_tasks for dep in self.dependencies)
    
    def get_input_data(self, workflow_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get input data for task execution."""
        input_data = self.parameters.copy()
        
        # Add dependency outputs to input
        for dep_id in self.dependencies:
            if dep_id in workflow_context:
                dep_result = workflow_context[dep_id]
                if isinstance(dep_result, TaskResult) and dep_result.output is not None:
                    input_data[f"{dep_id}_output"] = dep_result.output
        
        return input_data
    
    def execute(self, workflow_context: Dict[str, Any]) -> TaskResult:
        """Execute the task, always passing workflow_context as a keyword argument."""
        result = TaskResult(task_id=self.task_id, status=TaskStatus.RUNNING)
        result.start_time = datetime.now()

        try:
            # Get input data
            input_data = self.get_input_data(workflow_context)
            # Always pass workflow_context as a kwarg for dynamic context-aware tasks
            input_data["workflow_context"] = workflow_context

            # Execute function
            if asyncio.iscoroutinefunction(self.function):
                # Handle async functions
                loop = asyncio.get_event_loop()
                output = loop.run_until_complete(self.function(**input_data))
            else:
                # Handle sync functions
                output = self.function(**input_data)

            result.output = output
            result.status = TaskStatus.COMPLETED

        except Exception as e:
            result.error = e
            result.status = TaskStatus.FAILED
            logger.error(f"Task {self.name} failed: {e}")

        finally:
            result.end_time = datetime.now()
            if result.start_time:
                result.execution_time = (result.end_time - result.start_time).total_seconds()

        self.result = result
        self.status = result.status
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            'task_id': self.task_id,
            'name': self.name,
            'dependencies': self.dependencies,
            'parameters': self.parameters,
            'timeout': self.timeout,
            'retry_policy': self.retry_policy,
            'resources': self.resources,
            'metadata': self.metadata,
            'status': self.status.value,
            'created_at': self.created_at.isoformat()
        }


class Workflow:
    """Represents a workflow containing multiple tasks."""
    
    def __init__(
        self,
        name: str,
        workflow_id: Optional[str] = None,
        description: str = "",
        tasks: Optional[List[Task]] = None,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.workflow_id = workflow_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.tasks: Dict[str, Task] = {}
        self.config = config or {}
        self.metadata = metadata or {}
        
        # Runtime state
        self.status = WorkflowStatus.CREATED
        self.result: Optional[WorkflowResult] = None
        self.created_at = datetime.now()
        self.context: Dict[str, Any] = {}
        
        # Add initial tasks
        if tasks:
            for task in tasks:
                self.add_task(task)
    
    def add_task(self, task: Task) -> None:
        """Add a task to the workflow."""
        self.tasks[task.task_id] = task
        logger.debug(f"Added task {task.name} to workflow {self.name}")
    
    def remove_task(self, task_id: str) -> None:
        """Remove a task from the workflow."""
        if task_id in self.tasks:
            del self.tasks[task_id]
            # Remove dependencies on this task
            for task in self.tasks.values():
                if task_id in task.dependencies:
                    task.dependencies.remove(task_id)
            logger.debug(f"Removed task {task_id} from workflow {self.name}")
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self.tasks.get(task_id)
    
    def get_ready_tasks(self, completed_tasks: Set[str]) -> List[Task]:
        """Get tasks that are ready to execute."""
        ready_tasks = []
        for task in self.tasks.values():
            if (task.status == TaskStatus.PENDING and 
                task.can_execute(completed_tasks)):
                ready_tasks.append(task)
        return ready_tasks
    
    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Get the dependency graph of tasks."""
        graph = {}
        for task_id, task in self.tasks.items():
            graph[task_id] = task.dependencies.copy()
        return graph
    
    def validate(self) -> List[str]:
        """Validate the workflow for issues."""
        issues = []
        
        # Check for circular dependencies
        if self._has_circular_dependencies():
            issues.append("Workflow has circular dependencies")
        
        # Check for missing dependencies
        all_task_ids = set(self.tasks.keys())
        for task in self.tasks.values():
            for dep in task.dependencies:
                if dep not in all_task_ids:
                    issues.append(f"Task {task.name} has missing dependency: {dep}")
        
        # Check for orphaned tasks (no path to execution)
        root_tasks = [t for t in self.tasks.values() if not t.dependencies]
        if not root_tasks and self.tasks:
            issues.append("No root tasks found (all tasks have dependencies)")
        
        return issues
    
    def _has_circular_dependencies(self) -> bool:
        """Check for circular dependencies using DFS."""
        graph = self.get_dependency_graph()
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for task_id in graph:
            if task_id not in visited:
                if has_cycle(task_id):
                    return True
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow to dictionary."""
        return {
            'workflow_id': self.workflow_id,
            'name': self.name,
            'description': self.description,
            'tasks': {task_id: task.to_dict() for task_id, task in self.tasks.items()},
            'config': self.config,
            'metadata': self.metadata,
            'status': self.status.value,
            'created_at': self.created_at.isoformat()
        }
    
    def save(self, file_path: str) -> None:
        """Save workflow to file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved workflow {self.name} to {file_path}")


class WorkflowEngine:
    """Main workflow engine for orchestrating workflow execution."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.workflows: Dict[str, Workflow] = {}
        self.execution_history: List[WorkflowResult] = []
        
        # Configure logging
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = self.config.get('log_level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def register_workflow(self, workflow: Workflow) -> None:
        """Register a workflow with the engine."""
        self.workflows[workflow.workflow_id] = workflow
        logger.info(f"Registered workflow: {workflow.name}")
    
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get a registered workflow."""
        return self.workflows.get(workflow_id)
    
    def run(
        self, 
        workflow: Union[Workflow, str], 
        executor_type: str = "parallel",
        **kwargs
    ) -> WorkflowResult:
        """Run a workflow."""
        # Get workflow object
        if isinstance(workflow, str):
            workflow_obj = self.get_workflow(workflow)
            if not workflow_obj:
                raise ValueError(f"Workflow not found: {workflow}")
        else:
            workflow_obj = workflow
        
        # Validate workflow
        issues = workflow_obj.validate()
        if issues:
            raise ValueError(f"Workflow validation failed: {issues}")
        
        # Import executor (avoid circular imports)
        from .execution import WorkflowExecutor
        
        # Create and run executor
        executor = WorkflowExecutor(engine=self, executor_type=executor_type)
        result = executor.execute(workflow_obj, **kwargs)
        
        # Store execution history
        self.execution_history.append(result)
        
        logger.info(f"Workflow {workflow_obj.name} completed with status: {result.status}")
        return result
    
    def get_execution_history(self, workflow_id: Optional[str] = None) -> List[WorkflowResult]:
        """Get execution history, optionally filtered by workflow ID."""
        if workflow_id:
            return [result for result in self.execution_history 
                   if result.workflow_id == workflow_id]
        return self.execution_history.copy()
    
    def load_workflow(self, file_path: str) -> Workflow:
        """Load workflow from file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Reconstruct workflow (simplified version)
        workflow = Workflow(
            name=data['name'],
            workflow_id=data['workflow_id'],
            description=data['description'],
            config=data['config'],
            metadata=data['metadata']
        )
        
        # Note: Task functions cannot be serialized, so this is a basic loader
        # In practice, you'd need a task registry or other mechanism
        logger.warning("Loaded workflow without task functions - register tasks separately")
        
        self.register_workflow(workflow)
        return workflow
