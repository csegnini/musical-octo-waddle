"""
Workflow Orchestration Package

This package provides advanced workflow orchestration capabilities for data science including:
- Multi-step workflow automation
- Intelligent task chaining and dependencies  
- Parallel and sequential execution
- Error recovery and retry mechanisms
- Workflow monitoring and logging
- Dynamic workflow modification
- Resource management and optimization
"""

from .core import WorkflowEngine, Task, Workflow, TaskResult, TaskStatus
from .scheduler import TaskScheduler, DependencyResolver, ExecutionPlan
from .execution import (
    WorkflowExecutor, 
    ParallelExecutor, 
    SequentialExecutor,
    TaskRunner
)
from .monitoring import WorkflowMonitor, ExecutionTracker, PerformanceAnalyzer
from .recovery import ErrorHandler, RetryPolicy, FallbackStrategy, RecoveryManager
from .utils import WorkflowBuilder, TaskFactory, WorkflowValidator, WorkflowSerializer, create_linear_workflow, create_parallel_workflow, create_map_reduce_workflow

# Version info
__version__ = "1.0.0"
__author__ = "Analytics Team"

# Main exports
__all__ = [
    # Core classes
    'WorkflowEngine',
    'Task', 
    'Workflow',
    'TaskResult',
    'TaskStatus',
    
    # Scheduling
    'TaskScheduler',
    'DependencyResolver',
    'ExecutionPlan',
    
    # Execution
    'WorkflowExecutor',
    'ParallelExecutor',
    'SequentialExecutor', 
    'TaskRunner',
    
    # Monitoring
    'WorkflowMonitor',
    'ExecutionTracker',
    'PerformanceAnalyzer',
    
    # Recovery
    'ErrorHandler',
    'RetryPolicy',
    'FallbackStrategy',
    'RecoveryManager',
    
    # Utilities
    'WorkflowBuilder',
    'TaskFactory',
    'WorkflowValidator',
    'WorkflowSerializer',
    
    # Utility functions
    'create_linear_workflow',
    'create_parallel_workflow', 
    'create_map_reduce_workflow',
    
    # Convenience functions
    'create_workflow',
    'run_workflow',
    'create_task',
    'chain_tasks',
    'parallel_tasks',
]

# Convenience functions
def create_workflow(name: str, description: str = "", **kwargs):
    """Quick workflow creation."""
    return Workflow(name=name, description=description, **kwargs)

def run_workflow(workflow, executor_type: str = "parallel", **kwargs):
    """Quick workflow execution."""
    engine = WorkflowEngine()
    return engine.run(workflow, executor_type=executor_type, **kwargs)

def create_task(name: str, func, **kwargs):
    """Quick task creation."""
    return Task(name=name, function=func, **kwargs)

def chain_tasks(*tasks):
    """Chain tasks in sequence."""
    builder = WorkflowBuilder("chained_workflow")
    return builder.chain(*tasks)

def parallel_tasks(*tasks):
    """Execute tasks in parallel."""
    builder = WorkflowBuilder("parallel_workflow")
    return builder.parallel(*tasks)
