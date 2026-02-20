"""
Task Scheduling Module

This module provides intelligent task scheduling capabilities including
dependency resolution, execution planning, and resource optimization.
"""

import heapq
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import networkx as nx
from collections import defaultdict, deque
import logging

from .core import Task, Workflow, TaskStatus

logger = logging.getLogger(__name__)


@dataclass
class ExecutionPlan:
    """Represents an execution plan for a workflow."""
    workflow_id: str
    execution_order: List[List[str]]  # Batches of tasks that can run in parallel
    dependency_graph: Dict[str, List[str]]
    critical_path: List[str]
    estimated_duration: float
    resource_requirements: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_batch_count(self) -> int:
        """Get the number of execution batches."""
        return len(self.execution_order)
    
    def get_max_parallelism(self) -> int:
        """Get the maximum number of tasks that can run in parallel."""
        return max(len(batch) for batch in self.execution_order) if self.execution_order else 0
    
    def get_task_level(self, task_id: str) -> Optional[int]:
        """Get the execution level (batch index) for a task."""
        for level, batch in enumerate(self.execution_order):
            if task_id in batch:
                return level
        return None


class DependencyResolver:
    """Resolves task dependencies and creates execution plans."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def resolve_dependencies(self, workflow: Workflow) -> ExecutionPlan:
        """Resolve task dependencies and create execution plan."""
        tasks = workflow.tasks
        
        # Build dependency graph
        dependency_graph = {}
        for task_id, task in tasks.items():
            dependency_graph[task_id] = task.dependencies.copy()
        
        # Validate dependencies
        self._validate_dependencies(dependency_graph, tasks)
        
        # Create execution order using topological sort with levels
        execution_order = self._create_execution_order(dependency_graph)
        
        # Find critical path
        critical_path = self._find_critical_path(dependency_graph, tasks)
        
        # Estimate duration
        estimated_duration = self._estimate_duration(critical_path, tasks)
        
        # Calculate resource requirements
        resource_requirements = self._calculate_resource_requirements(tasks)
        
        plan = ExecutionPlan(
            workflow_id=workflow.workflow_id,
            execution_order=execution_order,
            dependency_graph=dependency_graph,
            critical_path=critical_path,
            estimated_duration=estimated_duration,
            resource_requirements=resource_requirements
        )
        
        logger.info(f"Created execution plan for workflow {workflow.name}: "
                   f"{plan.get_batch_count()} batches, max parallelism: {plan.get_max_parallelism()}")
        
        return plan
    
    def _validate_dependencies(self, dependency_graph: Dict[str, List[str]], tasks: Dict[str, Task]) -> None:
        """Validate that all dependencies exist and there are no cycles."""
        all_task_ids = set(tasks.keys())
        
        # Check for missing dependencies
        for task_id, dependencies in dependency_graph.items():
            for dep in dependencies:
                if dep not in all_task_ids:
                    raise ValueError(f"Task {task_id} has missing dependency: {dep}")
        
        # Check for cycles using DFS
        if self._has_cycles(dependency_graph):
            raise ValueError("Circular dependencies detected in workflow")
    
    def _has_cycles(self, graph: Dict[str, List[str]]) -> bool:
        """Detect cycles in dependency graph using DFS."""
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {node: WHITE for node in graph}
        
        def has_cycle_util(node):
            if colors[node] == GRAY:
                return True
            if colors[node] == BLACK:
                return False
            
            colors[node] = GRAY
            for neighbor in graph.get(node, []):
                if has_cycle_util(neighbor):
                    return True
            colors[node] = BLACK
            return False
        
        for node in graph:
            if colors[node] == WHITE:
                if has_cycle_util(node):
                    return True
        return False
    
    def _create_execution_order(self, dependency_graph: Dict[str, List[str]]) -> List[List[str]]:
        """Create execution order using level-based topological sort (Kahn's algorithm)."""
        from collections import defaultdict, deque
        in_degree = defaultdict(int)
        for node in dependency_graph:
            in_degree[node] = 0
        for node, dependencies in dependency_graph.items():
            for dep in dependencies:
                in_degree[node] += 1

        execution_order = []
        queue = deque([node for node in dependency_graph if in_degree[node] == 0])

        while queue:
            current_level = []
            next_queue = deque()
            while queue:
                node = queue.popleft()
                current_level.append(node)
                for dependent in dependency_graph:
                    if node in dependency_graph[dependent]:
                        in_degree[dependent] -= 1
                        if in_degree[dependent] == 0:
                            next_queue.append(dependent)
            if current_level:
                execution_order.append(current_level)
            queue = next_queue
        return execution_order
    
    def _find_critical_path(self, dependency_graph: Dict[str, List[str]], tasks: Dict[str, Task]) -> List[str]:
        """Find the critical path (longest path) through the workflow."""
        # Estimate task durations
        task_durations = {}
        for task_id, task in tasks.items():
            # Use configured timeout or estimate based on task complexity
            duration = task.timeout or self._estimate_task_duration(task)
            task_durations[task_id] = duration
        
        # Use networkx for critical path calculation
        G = nx.DiGraph()
        
        # Add nodes with durations
        for task_id, duration in task_durations.items():
            G.add_node(task_id, duration=duration)
        
        # Add edges (dependencies)
        for task_id, dependencies in dependency_graph.items():
            for dep in dependencies:
                G.add_edge(dep, task_id)
        
        # Find longest path (critical path)
        try:
            # Calculate longest path using topological sort
            critical_path = self._longest_path(G, task_durations)
            return critical_path
        except Exception as e:
            logger.warning(f"Could not calculate critical path: {e}")
            return []
    
    def _longest_path(self, graph: nx.DiGraph, durations: Dict[str, float]) -> List[str]:
        """Calculate longest path in DAG."""
        # Topological sort
        topo_order = list(nx.topological_sort(graph))
        
        # Calculate longest distances
        distances = {node: 0.0 for node in graph.nodes()}
        predecessors = {node: None for node in graph.nodes()}
        
        for node in topo_order:
            for successor in graph.successors(node):
                new_distance = distances[node] + durations[node]
                if new_distance > distances[successor]:
                    distances[successor] = new_distance
                    predecessors[successor] = node
        
        # Find the node with maximum distance
        max_node = max(distances.items(), key=lambda x: x[1])[0]
        
        # Reconstruct path
        path = []
        current = max_node
        while current is not None:
            path.append(current)
            current = predecessors[current]
        
        return list(reversed(path))
    
    def _estimate_task_duration(self, task: Task) -> float:
        """Estimate task duration based on task characteristics."""
        # Default estimation logic - can be overridden
        base_duration = 10.0  # seconds
        
        # Adjust based on task complexity indicators
        if 'complexity' in task.metadata:
            complexity = task.metadata['complexity']
            if complexity == 'low':
                return base_duration * 0.5
            elif complexity == 'high':
                return base_duration * 3.0
        
        # Adjust based on resource requirements
        if task.resources:
            if task.resources.get('cpu_intensive', False):
                base_duration *= 2.0
            if task.resources.get('memory_intensive', False):
                base_duration *= 1.5
            if task.resources.get('io_intensive', False):
                base_duration *= 1.2
        
        return base_duration
    
    def _estimate_duration(self, critical_path: List[str], tasks: Dict[str, Task]) -> float:
        """Estimate total workflow duration based on critical path."""
        total_duration = 0.0
        for task_id in critical_path:
            if task_id in tasks:
                task = tasks[task_id]
                duration = task.timeout or self._estimate_task_duration(task)
                total_duration += duration
        return total_duration
    
    def _calculate_resource_requirements(self, tasks: Dict[str, Task]) -> Dict[str, Any]:
        """Calculate aggregate resource requirements."""
        requirements = {
            'max_cpu_cores': 0,
            'max_memory_mb': 0,
            'total_storage_mb': 0,
            'gpu_required': False,
            'network_intensive': False
        }
        
        for task in tasks.values():
            if task.resources:
                requirements['max_cpu_cores'] = max(
                    requirements['max_cpu_cores'],
                    task.resources.get('cpu_cores', 1)
                )
                requirements['max_memory_mb'] = max(
                    requirements['max_memory_mb'], 
                    task.resources.get('memory_mb', 512)
                )
                requirements['total_storage_mb'] += task.resources.get('storage_mb', 0)
                
                if task.resources.get('gpu_required', False):
                    requirements['gpu_required'] = True
                if task.resources.get('network_intensive', False):
                    requirements['network_intensive'] = True
        
        return requirements


class TaskScheduler:
    """Intelligent task scheduler with resource optimization."""
    
    def __init__(self, max_concurrent_tasks: int = 10, resource_limits: Optional[Dict[str, Any]] = None):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.resource_limits = resource_limits or {}
        self.dependency_resolver = DependencyResolver()
        self.logger = logging.getLogger(__name__)
    
    def create_schedule(self, workflow: Workflow) -> ExecutionPlan:
        """Create an optimized execution schedule for the workflow."""
        # Get basic execution plan
        plan = self.dependency_resolver.resolve_dependencies(workflow)
        
        # Optimize plan based on resources and constraints
        optimized_plan = self._optimize_execution_plan(plan, workflow)
        
        logger.info(f"Created optimized schedule for workflow {workflow.name}")
        return optimized_plan
    
    def _optimize_execution_plan(self, plan: ExecutionPlan, workflow: Workflow) -> ExecutionPlan:
        """Optimize execution plan based on resource constraints."""
        optimized_order = []
        
        for batch in plan.execution_order:
            # Split large batches if they exceed concurrent task limit
            if len(batch) > self.max_concurrent_tasks:
                # Use priority-based splitting
                optimized_batches = self._split_batch_by_priority(batch, workflow.tasks)
                optimized_order.extend(optimized_batches)
            else:
                optimized_order.append(batch)
        
        # Update the plan
        plan.execution_order = optimized_order
        return plan
    
    def _split_batch_by_priority(self, batch: List[str], tasks: Dict[str, Task]) -> List[List[str]]:
        """Split a large batch into smaller batches based on task priority."""
        # Calculate task priorities
        task_priorities = []
        for task_id in batch:
            task = tasks[task_id]
            priority = self._calculate_task_priority(task)
            task_priorities.append((priority, task_id))
        
        # Sort by priority (higher first)
        task_priorities.sort(reverse=True)
        
        # Split into batches
        batches = []
        current_batch = []
        
        for priority, task_id in task_priorities:
            current_batch.append(task_id)
            
            if len(current_batch) >= self.max_concurrent_tasks:
                batches.append(current_batch)
                current_batch = []
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _calculate_task_priority(self, task: Task) -> float:
        """Calculate task priority for scheduling."""
        priority = 1.0
        
        # Higher priority for tasks with more dependents
        # This would require reverse dependency tracking
        
        # Adjust based on resource requirements
        if task.resources:
            if task.resources.get('cpu_intensive', False):
                priority += 0.5
            if task.resources.get('memory_intensive', False):
                priority += 0.3
            if task.resources.get('gpu_required', False):
                priority += 1.0
        
        # Adjust based on estimated duration
        if task.timeout:
            # Shorter tasks get slightly higher priority
            priority += max(0, (60 - task.timeout) / 100)
        
        # Check metadata for explicit priority
        if 'priority' in task.metadata:
            priority += task.metadata['priority']
        
        return priority
    
    def get_next_tasks(self, workflow: Workflow, completed_tasks: Set[str], running_tasks: Set[str]) -> List[Task]:
        """Get the next batch of tasks that can be scheduled."""
        ready_tasks = workflow.get_ready_tasks(completed_tasks)
        
        # Filter out already running tasks
        ready_tasks = [task for task in ready_tasks if task.task_id not in running_tasks]
        
        # Limit by max concurrent tasks
        available_slots = self.max_concurrent_tasks - len(running_tasks)
        if available_slots <= 0:
            return []
        
        # Apply resource constraints
        schedulable_tasks = self._apply_resource_constraints(ready_tasks[:available_slots])
        
        return schedulable_tasks
    
    def _apply_resource_constraints(self, tasks: List[Task]) -> List[Task]:
        """Apply resource constraints to filter schedulable tasks."""
        if not self.resource_limits:
            return tasks
        
        schedulable = []
        current_cpu = 0
        current_memory = 0
        
        max_cpu = self.resource_limits.get('max_cpu_cores', float('inf'))
        max_memory = self.resource_limits.get('max_memory_mb', float('inf'))
        
        for task in tasks:
            task_cpu = task.resources.get('cpu_cores', 1)
            task_memory = task.resources.get('memory_mb', 512)
            
            if (current_cpu + task_cpu <= max_cpu and 
                current_memory + task_memory <= max_memory):
                schedulable.append(task)
                current_cpu += task_cpu
                current_memory += task_memory
            else:
                logger.debug(f"Task {task.name} skipped due to resource constraints")
        
        return schedulable
