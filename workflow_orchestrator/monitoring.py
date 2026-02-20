"""
Workflow Monitoring Module

This module provides comprehensive monitoring and analytics capabilities
for workflow execution including real-time tracking, performance analysis,
and reporting.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import json
from pathlib import Path
import logging

from .core import Workflow, Task, TaskResult, WorkflowResult, TaskStatus, WorkflowStatus

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for tasks and workflows."""
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    max_execution_time: float = 0.0
    success_rate: float = 0.0
    failure_rate: float = 0.0
    throughput: float = 0.0  # tasks per second
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    
    def update_with_result(self, result: TaskResult):
        """Update metrics with a new task result."""
        if result.execution_time:
            self.total_execution_time += result.execution_time
            self.min_execution_time = min(self.min_execution_time, result.execution_time)
            self.max_execution_time = max(self.max_execution_time, result.execution_time)


@dataclass
class ExecutionEvent:
    """Represents an execution event for monitoring."""
    timestamp: datetime
    event_type: str  # started, completed, failed, etc.
    workflow_id: str
    task_id: Optional[str] = None
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExecutionTracker:
    """Tracks workflow and task execution in real-time."""
    
    def __init__(self, max_events: int = 10000):
        self.max_events = max_events
        self.events: deque = deque(maxlen=max_events)
        self.active_workflows: Dict[str, WorkflowResult] = {}
        self.active_tasks: Dict[str, TaskResult] = {}
        self.workflow_metrics: Dict[str, PerformanceMetrics] = defaultdict(PerformanceMetrics)
        self.task_metrics: Dict[str, PerformanceMetrics] = defaultdict(PerformanceMetrics)
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def track_workflow_start(self, workflow: Workflow):
        """Track workflow start."""
        with self._lock:
            result = WorkflowResult(
                workflow_id=workflow.workflow_id,
                status=WorkflowStatus.RUNNING,
                start_time=datetime.now()
            )
            self.active_workflows[workflow.workflow_id] = result
            
            event = ExecutionEvent(
                timestamp=datetime.now(),
                event_type="workflow_started",
                workflow_id=workflow.workflow_id,
                message=f"Workflow '{workflow.name}' started"
            )
            self.events.append(event)
            self.logger.info(f"Tracking started for workflow: {workflow.name}")
    
    def track_workflow_end(self, workflow_result: WorkflowResult):
        """Track workflow completion."""
        with self._lock:
            if workflow_result.workflow_id in self.active_workflows:
                self.active_workflows[workflow_result.workflow_id] = workflow_result
                
                # Update metrics
                metrics = self.workflow_metrics[workflow_result.workflow_id]
                if workflow_result.duration:
                    metrics.total_execution_time = workflow_result.duration.total_seconds()
                
                # Calculate success rate
                total_tasks = len(workflow_result.task_results)
                successful_tasks = len(workflow_result.successful_tasks)
                if total_tasks > 0:
                    metrics.success_rate = successful_tasks / total_tasks
                    metrics.failure_rate = 1.0 - metrics.success_rate
                
                event = ExecutionEvent(
                    timestamp=datetime.now(),
                    event_type="workflow_completed",
                    workflow_id=workflow_result.workflow_id,
                    message=f"Workflow completed with status: {workflow_result.status.value}",
                    metadata={
                        "status": workflow_result.status.value,
                        "duration": workflow_result.duration.total_seconds() if workflow_result.duration else 0,
                        "success_rate": metrics.success_rate
                    }
                )
                self.events.append(event)
    
    def track_task_start(self, task: Task, workflow_id: str):
        """Track task start."""
        with self._lock:
            result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.RUNNING,
                start_time=datetime.now()
            )
            self.active_tasks[task.task_id] = result
            
            event = ExecutionEvent(
                timestamp=datetime.now(),
                event_type="task_started",
                workflow_id=workflow_id,
                task_id=task.task_id,
                message=f"Task '{task.name}' started"
            )
            self.events.append(event)
    
    def track_task_end(self, task_result: TaskResult, workflow_id: str):
        """Track task completion."""
        with self._lock:
            if task_result.task_id in self.active_tasks:
                self.active_tasks[task_result.task_id] = task_result
                
                # Update metrics
                metrics = self.task_metrics[task_result.task_id]
                metrics.update_with_result(task_result)
                
                event = ExecutionEvent(
                    timestamp=datetime.now(),
                    event_type="task_completed",
                    workflow_id=workflow_id,
                    task_id=task_result.task_id,
                    message=f"Task completed with status: {task_result.status.value}",
                    metadata={
                        "status": task_result.status.value,
                        "execution_time": task_result.execution_time,
                        "retry_count": task_result.retry_count
                    }
                )
                self.events.append(event)
    
    def get_active_workflows(self) -> Dict[str, WorkflowResult]:
        """Get currently active workflows."""
        with self._lock:
            return self.active_workflows.copy()
    
    def get_active_tasks(self) -> Dict[str, TaskResult]:
        """Get currently active tasks."""
        with self._lock:
            return self.active_tasks.copy()
    
    def get_recent_events(self, count: int = 100) -> List[ExecutionEvent]:
        """Get recent execution events."""
        with self._lock:
            return list(self.events)[-count:]
    
    def get_workflow_metrics(self, workflow_id: str) -> Optional[PerformanceMetrics]:
        """Get performance metrics for a workflow."""
        return self.workflow_metrics.get(workflow_id)
    
    def clear_completed(self):
        """Clear completed workflows and tasks from active tracking."""
        with self._lock:
            # Remove completed workflows
            completed_workflows = [
                wf_id for wf_id, result in self.active_workflows.items()
                if result.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]
            ]
            for wf_id in completed_workflows:
                del self.active_workflows[wf_id]
            
            # Remove completed tasks
            completed_tasks = [
                task_id for task_id, result in self.active_tasks.items()
                if result.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
            ]
            for task_id in completed_tasks:
                del self.active_tasks[task_id]


class PerformanceAnalyzer:
    """Analyzes workflow and task performance patterns."""
    
    def __init__(self, tracker: ExecutionTracker):
        self.tracker = tracker
        self.logger = logging.getLogger(__name__)
    
    def analyze_workflow_performance(self, workflow_id: str) -> Dict[str, Any]:
        """Analyze performance of a specific workflow."""
        metrics = self.tracker.get_workflow_metrics(workflow_id)
        if not metrics:
            return {}
        
        # Get workflow events
        workflow_events = [
            event for event in self.tracker.events
            if event.workflow_id == workflow_id
        ]
        
        analysis = {
            "workflow_id": workflow_id,
            "total_execution_time": metrics.total_execution_time,
            "success_rate": metrics.success_rate,
            "failure_rate": metrics.failure_rate,
            "event_count": len(workflow_events),
            "performance_grade": self._calculate_performance_grade(metrics)
        }
        
        return analysis
    
    def analyze_task_performance(self, task_name_pattern: Optional[str] = None) -> Dict[str, Any]:
        """Analyze performance across tasks."""
        task_performances = []
        
        for task_id, metrics in self.tracker.task_metrics.items():
            if task_name_pattern and task_name_pattern not in task_id:
                continue
            
            task_performances.append({
                "task_id": task_id,
                "total_execution_time": metrics.total_execution_time,
                "min_execution_time": metrics.min_execution_time if metrics.min_execution_time != float('inf') else 0,
                "max_execution_time": metrics.max_execution_time,
                "performance_grade": self._calculate_performance_grade(metrics)
            })
        
        # Calculate aggregate statistics
        if task_performances:
            execution_times = [tp["total_execution_time"] for tp in task_performances]
            analysis = {
                "task_count": len(task_performances),
                "average_execution_time": statistics.mean(execution_times) if execution_times else 0,
                "median_execution_time": statistics.median(execution_times) if execution_times else 0,
                "std_execution_time": statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                "tasks": task_performances
            }
        else:
            analysis = {
                "task_count": 0,
                "tasks": []
            }
        
        return analysis
    
    def identify_bottlenecks(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks in a workflow."""
        bottlenecks = []
        
        # Get workflow events
        task_events = [
            event for event in self.tracker.events
            if event.workflow_id == workflow_id and event.task_id
        ]
        
        # Group events by task
        task_event_groups = defaultdict(list)
        for event in task_events:
            task_event_groups[event.task_id].append(event)
        
        # Analyze each task
        task_durations = []
        for task_id, events in task_event_groups.items():
            metrics = self.tracker.task_metrics.get(task_id)
            if metrics and metrics.total_execution_time > 0:
                task_durations.append((task_id, metrics.total_execution_time))
        
        if task_durations:
            # Sort by duration
            task_durations.sort(key=lambda x: x[1], reverse=True)
            
            # Calculate threshold for bottlenecks (top 20% or tasks > 2x average)
            avg_duration = sum(duration for _, duration in task_durations) / len(task_durations)
            threshold = max(avg_duration * 2, task_durations[max(1, len(task_durations) // 5)][1])
            
            for task_id, duration in task_durations:
                if duration >= threshold:
                    bottlenecks.append({
                        "task_id": task_id,
                        "execution_time": duration,
                        "severity": "high" if duration > avg_duration * 3 else "medium",
                        "recommendation": self._get_bottleneck_recommendation(duration, avg_duration)
                    })
        
        return bottlenecks
    
    def generate_performance_report(self, workflow_id: str) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        workflow_analysis = self.analyze_workflow_performance(workflow_id)
        bottlenecks = self.identify_bottlenecks(workflow_id)
        
        # Get recent events for timeline
        workflow_events = [
            {
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type,
                "task_id": event.task_id,
                "message": event.message
            }
            for event in self.tracker.events
            if event.workflow_id == workflow_id
        ]
        
        report = {
            "workflow_id": workflow_id,
            "generated_at": datetime.now().isoformat(),
            "performance_analysis": workflow_analysis,
            "bottlenecks": bottlenecks,
            "event_timeline": workflow_events[-50:],  # Last 50 events
            "recommendations": self._generate_recommendations(workflow_analysis, bottlenecks)
        }
        
        return report
    
    def _calculate_performance_grade(self, metrics: PerformanceMetrics) -> str:
        """Calculate performance grade based on metrics."""
        score = 0
        
        # Success rate weight: 40%
        score += metrics.success_rate * 40
        
        # Execution time weight: 30% (lower is better, assume good baseline)
        if metrics.total_execution_time > 0:
            # Simplified scoring - could be more sophisticated
            time_score = max(0, 30 - (metrics.total_execution_time / 10))
            score += min(30, time_score)
        
        # Resource utilization weight: 30%
        if metrics.resource_utilization:
            avg_utilization = sum(metrics.resource_utilization.values()) / len(metrics.resource_utilization)
            # Optimal utilization around 70-80%
            utilization_score = 30 - abs(75 - avg_utilization * 100) / 5
            score += max(0, utilization_score)
        else:
            score += 20  # Default if no resource data
        
        # Convert to grade
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def _get_bottleneck_recommendation(self, duration: float, avg_duration: float) -> str:
        """Get recommendation for addressing bottleneck."""
        ratio = duration / avg_duration if avg_duration > 0 else 1
        
        if ratio > 5:
            return "Critical bottleneck - consider breaking into smaller tasks or optimizing algorithm"
        elif ratio > 3:
            return "Significant bottleneck - review implementation and consider parallel processing"
        elif ratio > 2:
            return "Minor bottleneck - monitor and consider optimization if pattern persists"
        else:
            return "Within acceptable range"
    
    def _generate_recommendations(self, analysis: Dict[str, Any], bottlenecks: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Success rate recommendations
        success_rate = analysis.get("success_rate", 1.0)
        if success_rate < 0.8:
            recommendations.append("Low success rate detected - review error handling and task dependencies")
        elif success_rate < 0.95:
            recommendations.append("Consider adding retry policies for improved reliability")
        
        # Performance grade recommendations
        grade = analysis.get("performance_grade", "A")
        if grade in ["D", "F"]:
            recommendations.append("Poor performance grade - comprehensive optimization needed")
        elif grade == "C":
            recommendations.append("Average performance - identify and address bottlenecks")
        
        # Bottleneck recommendations
        high_severity_bottlenecks = [b for b in bottlenecks if b.get("severity") == "high"]
        if high_severity_bottlenecks:
            recommendations.append(f"Address {len(high_severity_bottlenecks)} critical bottlenecks identified")
        
        if not recommendations:
            recommendations.append("Performance is within acceptable range - continue monitoring")
        
        return recommendations


class WorkflowMonitor:
    """Comprehensive workflow monitoring system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.tracker = ExecutionTracker(
            max_events=self.config.get('max_events', 10000)
        )
        self.analyzer = PerformanceAnalyzer(self.tracker)
        self.callbacks: List[Callable] = []
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        self.logger = logging.getLogger(__name__)
    
    def add_callback(self, callback: Callable[[ExecutionEvent], None]):
        """Add a callback for monitoring events."""
        self.callbacks.append(callback)
    
    def start_monitoring(self, interval: float = 5.0):
        """Start continuous monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("Workflow monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info("Workflow monitoring stopped")
    
    def _monitoring_loop(self, interval: float):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Trigger callbacks for recent events
                recent_events = self.tracker.get_recent_events(10)
                for event in recent_events:
                    for callback in self.callbacks:
                        try:
                            callback(event)
                        except Exception as e:
                            self.logger.error(f"Monitoring callback error: {e}")
                
                # Clean up completed items periodically
                self.tracker.clear_completed()
                
                time.sleep(interval)
            
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(interval)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        active_workflows = self.tracker.get_active_workflows()
        active_tasks = self.tracker.get_active_tasks()
        recent_events = self.tracker.get_recent_events(20)
        
        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "active_workflows": len(active_workflows),
            "active_tasks": len(active_tasks),
            "recent_events": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.event_type,
                    "workflow_id": event.workflow_id,
                    "message": event.message
                }
                for event in recent_events
            ],
            "workflow_status": {
                workflow_id: result.status.value
                for workflow_id, result in active_workflows.items()
            }
        }
        
        return dashboard_data
    
    def export_metrics(self, file_path: str):
        """Export metrics to file."""
        metrics_data = {
            "exported_at": datetime.now().isoformat(),
            "workflow_metrics": {
                wf_id: {
                    "total_execution_time": metrics.total_execution_time,
                    "success_rate": metrics.success_rate,
                    "failure_rate": metrics.failure_rate
                }
                for wf_id, metrics in self.tracker.workflow_metrics.items()
            },
            "task_metrics": {
                task_id: {
                    "total_execution_time": metrics.total_execution_time,
                    "min_execution_time": metrics.min_execution_time if metrics.min_execution_time != float('inf') else 0,
                    "max_execution_time": metrics.max_execution_time
                }
                for task_id, metrics in self.tracker.task_metrics.items()
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        self.logger.info(f"Metrics exported to {file_path}")
