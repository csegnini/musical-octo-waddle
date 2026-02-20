"""
Workflow Execution Module

This module provides different execution strategies for workflows including
parallel, sequential, and hybrid execution with resource management.
"""

import asyncio
import threading
import multiprocessing as mp
from typing import Dict, List, Set, Optional, Any, Callable, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
from datetime import datetime
import time
import logging
from queue import Queue, Empty
import signal
import sys

from .core import Task, Workflow, TaskResult, WorkflowResult, TaskStatus, WorkflowStatus
from .scheduler import TaskScheduler, ExecutionPlan

logger = logging.getLogger(__name__)


class TaskRunner:
    """Handles individual task execution with timeout and resource management."""
    
    def __init__(self, timeout_handler: bool = True):
        self.timeout_handler = timeout_handler
        self.logger = logging.getLogger(__name__)
    
    def run_task(self, task: Task, workflow_context: Dict[str, Any]) -> TaskResult:
        """Execute a single task with timeout and error handling."""
        result = TaskResult(task_id=task.task_id, status=TaskStatus.RUNNING)
        result.start_time = datetime.now()
        
        try:
            if task.timeout and self.timeout_handler:
                # Run with timeout
                result = self._run_with_timeout(task, workflow_context, task.timeout)
            else:
                # Run without timeout
                result = task.execute(workflow_context)
            
        except Exception as e:
            result.error = e
            result.status = TaskStatus.FAILED
            self.logger.error(f"Task {task.name} execution failed: {e}")
        
        finally:
            if not result.end_time:
                result.end_time = datetime.now()
            if result.start_time and result.end_time:
                result.execution_time = (result.end_time - result.start_time).total_seconds()
        
        return result
    
    def _run_with_timeout(self, task: Task, workflow_context: Dict[str, Any], timeout: float) -> TaskResult:
        """Run task with timeout using threading."""
        result_queue = Queue()
        
        def target():
            try:
                result = task.execute(workflow_context)
                result_queue.put(result)
            except Exception as e:
                error_result = TaskResult(task_id=task.task_id, status=TaskStatus.FAILED)
                error_result.error = e
                result_queue.put(error_result)
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        
        try:
            result = result_queue.get(timeout=timeout)
            return result
        except Empty:
            # Timeout occurred
            self.logger.warning(f"Task {task.name} timed out after {timeout} seconds")
            timeout_result = TaskResult(task_id=task.task_id, status=TaskStatus.FAILED)
            timeout_result.error = TimeoutError(f"Task timed out after {timeout} seconds")
            return timeout_result


class SequentialExecutor:
    """Executes workflow tasks sequentially."""
    
    def __init__(self, task_runner: Optional[TaskRunner] = None):
        self.task_runner = task_runner or TaskRunner()
        self.logger = logging.getLogger(__name__)
    
    def execute(self, workflow: Workflow, scheduler: TaskScheduler) -> WorkflowResult:
        """Execute workflow sequentially."""
        result = WorkflowResult(workflow_id=workflow.workflow_id, status=WorkflowStatus.RUNNING)
        result.start_time = datetime.now()
        
        completed_tasks = set()
        workflow_context = workflow.context.copy()
        
        try:
            # Get execution plan
            plan = scheduler.create_schedule(workflow)
            
            # Execute tasks in order
            for batch in plan.execution_order:
                for task_id in batch:
                    task = workflow.get_task(task_id)
                    if not task:
                        continue
                    
                    self.logger.info(f"Executing task: {task.name}")
                    
                    # Execute task
                    task_result = self.task_runner.run_task(task, workflow_context)
                    result.task_results[task_id] = task_result
                    
                    # Update context and completed tasks
                    if task_result.is_successful:
                        completed_tasks.add(task_id)
                        workflow_context[task_id] = task_result
                    else:
                        # Task failed - decide whether to continue or stop
                        if self._should_stop_on_failure(task, workflow):
                            result.status = WorkflowStatus.FAILED
                            result.error = task_result.error
                            break
            
            # Determine final status
            if result.status == WorkflowStatus.RUNNING:
                if len(completed_tasks) == len(workflow.tasks):
                    result.status = WorkflowStatus.COMPLETED
                else:
                    result.status = WorkflowStatus.FAILED
        
        except Exception as e:
            result.status = WorkflowStatus.FAILED
            result.error = e
            self.logger.error(f"Workflow execution failed: {e}")
        
        finally:
            result.end_time = datetime.now()
        
        return result
    
    def _should_stop_on_failure(self, task: Task, workflow: Workflow) -> bool:
        """Determine if workflow should stop when a task fails."""
        # Check task-specific configuration
        if 'continue_on_failure' in task.metadata:
            return not task.metadata['continue_on_failure']
        
        # Check workflow-specific configuration
        if 'continue_on_failure' in workflow.config:
            return not workflow.config['continue_on_failure']
        
        # Default: stop on failure
        return True


class ParallelExecutor:
    """Executes workflow tasks in parallel using thread or process pools."""
    
    def __init__(
        self, 
        max_workers: Optional[int] = None,
        execution_mode: str = "thread",  # "thread" or "process"
        task_runner: Optional[TaskRunner] = None
    ):
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        self.execution_mode = execution_mode
        self.task_runner = task_runner or TaskRunner()
        self.logger = logging.getLogger(__name__)
    
    def execute(self, workflow: Workflow, scheduler: TaskScheduler) -> WorkflowResult:
        """Execute workflow with parallel task execution."""
        result = WorkflowResult(workflow_id=workflow.workflow_id, status=WorkflowStatus.RUNNING)
        result.start_time = datetime.now()
        
        completed_tasks = set()
        running_tasks = set()
        workflow_context = workflow.context.copy()
        
        try:
            # Get execution plan
            plan = scheduler.create_schedule(workflow)
            
            # Choose executor
            executor_class = ThreadPoolExecutor if self.execution_mode == "thread" else ProcessPoolExecutor
            
            with executor_class(max_workers=self.max_workers) as executor:
                # Process each batch
                for batch_index, batch in enumerate(plan.execution_order):
                    self.logger.info(f"Processing batch {batch_index + 1}/{len(plan.execution_order)} "
                                   f"with {len(batch)} tasks")

                    # Submit tasks in current batch
                    futures = {}
                    task_id_to_future = {}
                    for task_id in batch:
                        task = workflow.get_task(task_id)
                        if not task:
                            continue
                        # Use the shared workflow_context for all tasks in the batch
                        future = executor.submit(self.task_runner.run_task, task, workflow_context)
                        futures[future] = (task_id, task)
                        task_id_to_future[task_id] = future
                        running_tasks.add(task_id)

                    # Wait for batch completion, updating context as each finishes
                    batch_failed = False
                    completed_in_batch = set()
                    for future in as_completed(futures):
                        task_id, task = futures[future]
                        running_tasks.discard(task_id)

                        try:
                            task_result = future.result()
                            result.task_results[task_id] = task_result

                            if task_result.is_successful:
                                completed_tasks.add(task_id)
                                completed_in_batch.add(task_id)
                                workflow_context[task_id] = task_result
                                self.logger.info(f"Task {task.name} completed successfully")
                            else:
                                self.logger.error(f"Task {task.name} failed: {task_result.error}")
                                if self._should_stop_on_failure(task, workflow):
                                    batch_failed = True
                                    break

                        except Exception as e:
                            self.logger.error(f"Task {task_id} execution error: {e}")
                            error_result = TaskResult(task_id=task_id, status=TaskStatus.FAILED)
                            error_result.error = e
                            result.task_results[task_id] = error_result

                            if self._should_stop_on_failure(task, workflow):
                                batch_failed = True
                                break

                    # Stop if batch failed and we should stop on failure
                    if batch_failed:
                        result.status = WorkflowStatus.FAILED
                        break
            
            # Determine final status
            if result.status == WorkflowStatus.RUNNING:
                if len(completed_tasks) == len(workflow.tasks):
                    result.status = WorkflowStatus.COMPLETED
                else:
                    result.status = WorkflowStatus.FAILED
        
        except Exception as e:
            result.status = WorkflowStatus.FAILED
            result.error = e
            self.logger.error(f"Parallel workflow execution failed: {e}")
        
        finally:
            result.end_time = datetime.now()
        
        return result
    
    def _should_stop_on_failure(self, task: Task, workflow: Workflow) -> bool:
        """Determine if workflow should stop when a task fails."""
        # Check task-specific configuration
        if 'continue_on_failure' in task.metadata:
            return not task.metadata['continue_on_failure']
        
        # Check workflow-specific configuration
        if 'continue_on_failure' in workflow.config:
            return not workflow.config['continue_on_failure']
        
        # Default: stop on failure
        return True


class HybridExecutor:
    """Hybrid executor that uses both sequential and parallel execution strategies."""
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        parallel_threshold: int = 2,  # Minimum tasks in batch for parallel execution
        task_runner: Optional[TaskRunner] = None
    ):
        self.parallel_executor = ParallelExecutor(max_workers=max_workers, task_runner=task_runner)
        self.sequential_executor = SequentialExecutor(task_runner=task_runner)
        self.parallel_threshold = parallel_threshold
        self.logger = logging.getLogger(__name__)
    
    def execute(self, workflow: Workflow, scheduler: TaskScheduler) -> WorkflowResult:
        """Execute workflow using hybrid strategy."""
        result = WorkflowResult(workflow_id=workflow.workflow_id, status=WorkflowStatus.RUNNING)
        result.start_time = datetime.now()
        
        completed_tasks = set()
        workflow_context = workflow.context.copy()
        
        try:
            # Get execution plan
            plan = scheduler.create_schedule(workflow)
            
            # Process each batch with appropriate strategy
            for batch_index, batch in enumerate(plan.execution_order):
                if len(batch) >= self.parallel_threshold:
                    self.logger.info(f"Executing batch {batch_index + 1} in parallel ({len(batch)} tasks)")
                    strategy = "parallel"
                else:
                    self.logger.info(f"Executing batch {batch_index + 1} sequentially ({len(batch)} tasks)")
                    strategy = "sequential"
                
                # Create mini-workflow for this batch
                batch_workflow = Workflow(name=f"{workflow.name}_batch_{batch_index}")
                for task_id in batch:
                    task = workflow.get_task(task_id)
                    if task:
                        batch_workflow.add_task(task)
                
                batch_workflow.context = workflow_context.copy()
                
                # Execute batch
                if strategy == "parallel":
                    batch_result = self.parallel_executor.execute(batch_workflow, scheduler)
                else:
                    batch_result = self.sequential_executor.execute(batch_workflow, scheduler)
                
                # Merge results
                for task_id, task_result in batch_result.task_results.items():
                    result.task_results[task_id] = task_result
                    if task_result.is_successful:
                        completed_tasks.add(task_id)
                        workflow_context[task_id] = task_result
                
                # Check if we should continue
                if batch_result.status == WorkflowStatus.FAILED:
                    if self._should_stop_on_batch_failure(workflow):
                        result.status = WorkflowStatus.FAILED
                        result.error = batch_result.error
                        break
            
            # Determine final status
            if result.status == WorkflowStatus.RUNNING:
                if len(completed_tasks) == len(workflow.tasks):
                    result.status = WorkflowStatus.COMPLETED
                else:
                    result.status = WorkflowStatus.FAILED
        
        except Exception as e:
            result.status = WorkflowStatus.FAILED
            result.error = e
            self.logger.error(f"Hybrid workflow execution failed: {e}")
        
        finally:
            result.end_time = datetime.now()
        
        return result
    
    def _should_stop_on_batch_failure(self, workflow: Workflow) -> bool:
        """Determine if workflow should stop when a batch fails."""
        return workflow.config.get('stop_on_batch_failure', True)


class WorkflowExecutor:
    """Main workflow executor that coordinates different execution strategies."""
    
    def __init__(
        self, 
        engine=None,
        executor_type: str = "parallel",
        max_workers: Optional[int] = None,
        scheduler_config: Optional[Dict[str, Any]] = None
    ):
        self.engine = engine
        self.executor_type = executor_type
        self.max_workers = max_workers
        self.scheduler_config = scheduler_config or {}
        
        # Create scheduler
        self.scheduler = TaskScheduler(
            max_concurrent_tasks=self.scheduler_config.get('max_concurrent_tasks', 10),
            resource_limits=self.scheduler_config.get('resource_limits')
        )
        
        # Create task runner
        self.task_runner = TaskRunner(
            timeout_handler=self.scheduler_config.get('timeout_handler', True)
        )
        
        # Initialize executor
        self._initialize_executor()
        
        self.logger = logging.getLogger(__name__)
    
    def _initialize_executor(self):
        """Initialize the appropriate executor based on type."""
        if self.executor_type == "sequential":
            self.executor = SequentialExecutor(task_runner=self.task_runner)
        elif self.executor_type == "parallel":
            self.executor = ParallelExecutor(
                max_workers=self.max_workers,
                task_runner=self.task_runner
            )
        elif self.executor_type == "hybrid":
            self.executor = HybridExecutor(
                max_workers=self.max_workers,
                task_runner=self.task_runner
            )
        else:
            raise ValueError(f"Unknown executor type: {self.executor_type}")
    
    def execute(self, workflow: Workflow, **kwargs) -> WorkflowResult:
        """Execute a workflow using the configured strategy."""
        self.logger.info(f"Starting execution of workflow: {workflow.name} "
                        f"using {self.executor_type} executor")
        
        # Validate workflow before execution
        validation_issues = workflow.validate()
        if validation_issues:
            raise ValueError(f"Workflow validation failed: {validation_issues}")
        
        # Set workflow status
        workflow.status = WorkflowStatus.RUNNING
        
        try:
            # Execute workflow
            result = self.executor.execute(workflow, self.scheduler)
            
            # Update workflow status
            workflow.status = result.status
            workflow.result = result
            
            self.logger.info(f"Workflow {workflow.name} completed with status: {result.status}")
            return result
        
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            self.logger.error(f"Workflow execution failed: {e}")
            raise
    
    def execute_async(self, workflow: Workflow, **kwargs) -> Future:
        """Execute workflow asynchronously."""
        executor = ThreadPoolExecutor(max_workers=1)
        return executor.submit(self.execute, workflow, **kwargs)
