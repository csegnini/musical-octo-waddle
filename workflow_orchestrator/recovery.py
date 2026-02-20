"""
Workflow Recovery Module

This module provides comprehensive error handling, retry mechanisms,
and recovery strategies for workflow orchestration.
"""

import time
import random
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import traceback
from abc import ABC, abstractmethod

from .core import Task, Workflow, TaskResult, WorkflowResult, TaskStatus, WorkflowStatus

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Different retry strategies."""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    RANDOM_JITTER = "random_jitter"


class RecoveryAction(Enum):
    """Recovery actions that can be taken."""
    RETRY = "retry"
    SKIP = "skip"
    FALLBACK = "fallback"
    ABORT = "abort"
    MANUAL_INTERVENTION = "manual_intervention"


@dataclass
class RetryPolicy:
    """Defines retry behavior for tasks."""
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retry_on_exceptions: Optional[List[type]] = None
    stop_on_exceptions: Optional[List[type]] = None
    
    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """Determine if task should be retried."""
        if attempt >= self.max_attempts:
            return False
        
        # Check exception-based rules
        if self.stop_on_exceptions:
            for exc_type in self.stop_on_exceptions:
                if isinstance(exception, exc_type):
                    return False
        
        if self.retry_on_exceptions:
            for exc_type in self.retry_on_exceptions:
                if isinstance(exception, exc_type):
                    return True
            return False  # Only retry on specified exceptions
        
        return True  # Default: retry on any exception
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay before next retry attempt."""
        if self.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.base_delay
        elif self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.base_delay * (self.backoff_multiplier ** (attempt - 1))
        elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.base_delay * attempt
        elif self.strategy == RetryStrategy.RANDOM_JITTER:
            delay = self.base_delay + random.uniform(0, self.base_delay)
        else:
            delay = self.base_delay
        
        # Apply jitter if enabled
        if self.jitter and self.strategy != RetryStrategy.RANDOM_JITTER:
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        # Respect max delay
        return min(delay, self.max_delay)


class FallbackStrategy(ABC):
    """Abstract base class for fallback strategies."""
    
    @abstractmethod
    def execute(self, task: Task, workflow_context: Dict[str, Any], original_error: Exception) -> TaskResult:
        """Execute fallback strategy."""
        pass


class DefaultValueFallback(FallbackStrategy):
    """Fallback that returns a default value."""
    
    def __init__(self, default_value: Any):
        self.default_value = default_value
    
    def execute(self, task: Task, workflow_context: Dict[str, Any], original_error: Exception) -> TaskResult:
        """Return default value as task result."""
        result = TaskResult(task_id=task.task_id, status=TaskStatus.COMPLETED)
        result.output = self.default_value
        result.metadata['fallback_used'] = True
        result.metadata['original_error'] = str(original_error)
        logger.info(f"Using default value fallback for task {task.name}")
        return result


class AlternativeFunctionFallback(FallbackStrategy):
    """Fallback that executes an alternative function."""
    
    def __init__(self, fallback_function: Callable):
        self.fallback_function = fallback_function
    
    def execute(self, task: Task, workflow_context: Dict[str, Any], original_error: Exception) -> TaskResult:
        """Execute alternative function."""
        result = TaskResult(task_id=task.task_id, status=TaskStatus.RUNNING)
        result.start_time = datetime.now()
        
        try:
            input_data = task.get_input_data(workflow_context)
            input_data['original_error'] = original_error
            
            output = self.fallback_function(**input_data)
            result.output = output
            result.status = TaskStatus.COMPLETED
            result.metadata['fallback_used'] = True
            logger.info(f"Used alternative function fallback for task {task.name}")
            
        except Exception as e:
            result.error = e
            result.status = TaskStatus.FAILED
            logger.error(f"Fallback function also failed for task {task.name}: {e}")
        
        finally:
            result.end_time = datetime.now()
            if result.start_time:
                result.execution_time = (result.end_time - result.start_time).total_seconds()
        
        return result


class SkipTaskFallback(FallbackStrategy):
    """Fallback that marks task as skipped."""
    
    def execute(self, task: Task, workflow_context: Dict[str, Any], original_error: Exception) -> TaskResult:
        """Mark task as skipped."""
        result = TaskResult(task_id=task.task_id, status=TaskStatus.SKIPPED)
        result.metadata['fallback_used'] = True
        result.metadata['original_error'] = str(original_error)
        result.metadata['skip_reason'] = "Fallback strategy"
        logger.info(f"Skipping task {task.name} due to fallback strategy")
        return result


@dataclass
class RecoveryRule:
    """Defines recovery behavior for specific conditions."""
    condition: Callable[[Exception, Task], bool]
    action: RecoveryAction
    fallback_strategy: Optional[FallbackStrategy] = None
    retry_policy: Optional[RetryPolicy] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ErrorHandler:
    """Handles errors and determines recovery actions."""
    
    def __init__(self):
        self.recovery_rules: List[RecoveryRule] = []
        self.default_retry_policy = RetryPolicy()
        self.default_fallback = SkipTaskFallback()
        self.logger = logging.getLogger(__name__)
    
    def add_recovery_rule(self, rule: RecoveryRule):
        """Add a recovery rule."""
        self.recovery_rules.append(rule)
        self.logger.debug(f"Added recovery rule for action: {rule.action}")
    
    def handle_error(self, task: Task, exception: Exception, attempt: int) -> RecoveryAction:
        """Determine recovery action for an error."""
        # Check custom recovery rules
        for rule in self.recovery_rules:
            try:
                if rule.condition(exception, task):
                    self.logger.info(f"Applying recovery rule: {rule.action} for task {task.name}")
                    return rule.action
            except Exception as e:
                self.logger.warning(f"Error evaluating recovery rule: {e}")
        
        # Default behavior: retry based on task's retry policy
        retry_policy = self._get_retry_policy(task)
        if retry_policy.should_retry(attempt, exception):
            return RecoveryAction.RETRY
        else:
            return RecoveryAction.FALLBACK
    
    def get_fallback_strategy(self, task: Task, exception: Exception) -> FallbackStrategy:
        """Get appropriate fallback strategy for a task."""
        # Check recovery rules for specific fallback
        for rule in self.recovery_rules:
            try:
                if rule.condition(exception, task) and rule.fallback_strategy:
                    return rule.fallback_strategy
            except Exception as e:
                self.logger.warning(f"Error evaluating fallback rule: {e}")
        
        # Check task metadata for fallback configuration
        if 'fallback_strategy' in task.metadata:
            fallback_config = task.metadata['fallback_strategy']
            
            if fallback_config.get('type') == 'default_value':
                return DefaultValueFallback(fallback_config.get('value'))
            elif fallback_config.get('type') == 'skip':
                return SkipTaskFallback()
        
        # Return default fallback
        return self.default_fallback
    
    def _get_retry_policy(self, task: Task) -> RetryPolicy:
        """Get retry policy for a task."""
        if task.retry_policy:
            # Create RetryPolicy from task configuration
            return RetryPolicy(**task.retry_policy)
        return self.default_retry_policy


class RecoveryManager:
    """Manages the complete recovery process for failed tasks."""
    
    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        self.error_handler = error_handler or ErrorHandler()
        self.recovery_history: Dict[str, List[Dict[str, Any]]] = {}
        self.logger = logging.getLogger(__name__)
    
    def recover_task(
        self, 
        task: Task, 
        original_result: TaskResult, 
        workflow_context: Dict[str, Any],
        max_recovery_attempts: int = 3
    ) -> TaskResult:
        """Attempt to recover a failed task."""
        if not original_result.error:
            return original_result
        
        task_id = task.task_id
        recovery_attempts = self.recovery_history.get(task_id, [])
        
        # Check if we've exceeded max recovery attempts
        if len(recovery_attempts) >= max_recovery_attempts:
            self.logger.error(f"Max recovery attempts exceeded for task {task.name}")
            return original_result
        
        # Determine recovery action
        attempt_count = len(recovery_attempts) + 1
        recovery_action = self.error_handler.handle_error(task, original_result.error, attempt_count)
        
        # Record recovery attempt
        recovery_record = {
            'timestamp': datetime.now().isoformat(),
            'attempt': attempt_count,
            'action': recovery_action.value,
            'original_error': str(original_result.error),
            'original_error_type': type(original_result.error).__name__
        }
        
        if task_id not in self.recovery_history:
            self.recovery_history[task_id] = []
        self.recovery_history[task_id].append(recovery_record)
        
        # Execute recovery action
        if recovery_action == RecoveryAction.RETRY:
            return self._retry_task(task, workflow_context, attempt_count)
        elif recovery_action == RecoveryAction.FALLBACK:
            return self._fallback_task(task, workflow_context, original_result.error)
        elif recovery_action == RecoveryAction.SKIP:
            return self._skip_task(task, original_result.error)
        elif recovery_action == RecoveryAction.ABORT:
            self.logger.error(f"Aborting task {task.name} due to recovery policy")
            return original_result
        elif recovery_action == RecoveryAction.MANUAL_INTERVENTION:
            return self._request_manual_intervention(task, original_result.error)
        else:
            return original_result
    
    def _retry_task(self, task: Task, workflow_context: Dict[str, Any], attempt: int) -> TaskResult:
        """Retry task execution."""
        retry_policy = self.error_handler._get_retry_policy(task)
        delay = retry_policy.get_delay(attempt)
        
        if delay > 0:
            self.logger.info(f"Retrying task {task.name} in {delay:.2f} seconds (attempt {attempt})")
            time.sleep(delay)
        
        # Update task status for retry
        task.status = TaskStatus.RETRYING
        
        # Execute task again
        result = task.execute(workflow_context)
        result.retry_count = attempt
        
        # Update recovery history
        recovery_record = self.recovery_history[task.task_id][-1]
        recovery_record['retry_delay'] = delay
        recovery_record['retry_result'] = result.status.value
        
        self.logger.info(f"Retry attempt {attempt} for task {task.name}: {result.status.value}")
        return result
    
    def _fallback_task(self, task: Task, workflow_context: Dict[str, Any], original_error: Exception) -> TaskResult:
        """Execute fallback strategy for task."""
        fallback_strategy = self.error_handler.get_fallback_strategy(task, original_error)
        
        self.logger.info(f"Executing fallback strategy for task {task.name}: {type(fallback_strategy).__name__}")
        
        try:
            result = fallback_strategy.execute(task, workflow_context, original_error)
            
            # Update recovery history
            recovery_record = self.recovery_history[task.task_id][-1]
            recovery_record['fallback_strategy'] = type(fallback_strategy).__name__
            recovery_record['fallback_result'] = result.status.value
            
            return result
        
        except Exception as e:
            self.logger.error(f"Fallback strategy failed for task {task.name}: {e}")
            
            # Return failed result
            failed_result = TaskResult(task_id=task.task_id, status=TaskStatus.FAILED)
            failed_result.error = e
            return failed_result
    
    def _skip_task(self, task: Task, original_error: Exception) -> TaskResult:
        """Skip task due to recovery policy."""
        result = TaskResult(task_id=task.task_id, status=TaskStatus.SKIPPED)
        result.metadata['skip_reason'] = "Recovery policy"
        result.metadata['original_error'] = str(original_error)
        
        self.logger.info(f"Skipping task {task.name} due to recovery policy")
        return result
    
    def _request_manual_intervention(self, task: Task, original_error: Exception) -> TaskResult:
        """Request manual intervention for task."""
        self.logger.warning(f"Manual intervention requested for task {task.name}")
        
        # In a real implementation, this might:
        # - Send notifications to administrators
        # - Create tickets in issue tracking systems
        # - Pause workflow execution
        
        result = TaskResult(task_id=task.task_id, status=TaskStatus.FAILED)
        result.error = original_error
        result.metadata['manual_intervention_requested'] = True
        result.metadata['intervention_reason'] = str(original_error)
        
        return result
    
    def get_recovery_statistics(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Get recovery statistics."""
        if task_id:
            history = self.recovery_history.get(task_id, [])
            
            if not history:
                return {"task_id": task_id, "recovery_attempts": 0}
            
            actions = [record['action'] for record in history]
            return {
                "task_id": task_id,
                "recovery_attempts": len(history),
                "actions_taken": actions,
                "success_rate": sum(1 for record in history if record.get('retry_result') == 'completed') / len(history),
                "last_attempt": history[-1]['timestamp']
            }
        else:
            # Aggregate statistics
            total_attempts = sum(len(history) for history in self.recovery_history.values())
            successful_recoveries = 0
            
            for history in self.recovery_history.values():
                for record in history:
                    if record.get('retry_result') == 'completed' or record.get('fallback_result') == 'completed':
                        successful_recoveries += 1
            
            return {
                "total_recovery_attempts": total_attempts,
                "successful_recoveries": successful_recoveries,
                "recovery_success_rate": successful_recoveries / total_attempts if total_attempts > 0 else 0,
                "tasks_with_recovery": len(self.recovery_history)
            }
    
    def clear_recovery_history(self, task_id: Optional[str] = None):
        """Clear recovery history."""
        if task_id:
            self.recovery_history.pop(task_id, None)
        else:
            self.recovery_history.clear()
        
        self.logger.info(f"Cleared recovery history for {'task ' + task_id if task_id else 'all tasks'}")


# Predefined recovery rules for common scenarios
def create_timeout_recovery_rule(max_retries: int = 2) -> RecoveryRule:
    """Create recovery rule for timeout errors."""
    def timeout_condition(exception: Exception, task: Task) -> bool:
        return isinstance(exception, TimeoutError)
    
    return RecoveryRule(
        condition=timeout_condition,
        action=RecoveryAction.RETRY,
        retry_policy=RetryPolicy(max_attempts=max_retries, strategy=RetryStrategy.EXPONENTIAL_BACKOFF)
    )


def create_network_recovery_rule() -> RecoveryRule:
    """Create recovery rule for network-related errors."""
    def network_condition(exception: Exception, task: Task) -> bool:
        error_msg = str(exception).lower()
        network_indicators = ['connection', 'network', 'timeout', 'unreachable', 'dns']
        return any(indicator in error_msg for indicator in network_indicators)
    
    return RecoveryRule(
        condition=network_condition,
        action=RecoveryAction.RETRY,
        retry_policy=RetryPolicy(
            max_attempts=3,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay=2.0,
            max_delay=30.0
        )
    )


def create_memory_recovery_rule() -> RecoveryRule:
    """Create recovery rule for memory errors."""
    def memory_condition(exception: Exception, task: Task) -> bool:
        return isinstance(exception, MemoryError)
    
    return RecoveryRule(
        condition=memory_condition,
        action=RecoveryAction.FALLBACK,
        fallback_strategy=SkipTaskFallback()
    )


def create_file_not_found_recovery_rule(default_value: Any = None) -> RecoveryRule:
    """Create recovery rule for file not found errors."""
    def file_condition(exception: Exception, task: Task) -> bool:
        return isinstance(exception, FileNotFoundError)
    
    return RecoveryRule(
        condition=file_condition,
        action=RecoveryAction.FALLBACK,
        fallback_strategy=DefaultValueFallback(default_value)
    )
