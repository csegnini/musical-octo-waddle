"""
Workflow State Management Module

This module provides classes and enums for managing workflow state, status tracking,
and execution stage management in multi-agent systems.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from uuid import uuid4
import json

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Enumeration of possible workflow statuses."""
    INITIALIZED = "INITIALIZED"
    AWAITING_USER_INPUT = "AWAITING_USER_INPUT"
    PLANNING = "PLANNING"
    DATA_GATHERING = "DATA_GATHERING"
    DATA_PREPARATION = "DATA_PREPARATION"
    DATA_ANALYSIS = "DATA_ANALYSIS"
    MODELING = "MODELING"
    REPORTING = "REPORTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class StageStatus(Enum):
    """Enumeration of possible stage statuses."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    RETRYING = "RETRYING"


@dataclass
class ExecutionStep:
    """Represents a single execution step in the workflow."""
    step_id: str
    macro_process_name: str
    process_description: str
    responsible_agent_or_tool: str
    start_datetime_utc: Optional[str] = None
    end_datetime_utc: Optional[str] = None
    status: StageStatus = StageStatus.PENDING
    message: str = ""
    detailed_log_path: str = ""
    inputs_used: List[str] = field(default_factory=list)
    outputs_generated: List[str] = field(default_factory=list)
    error_details: Optional[Dict[str, Any]] = None
    duration_seconds: Optional[float] = None
    decision_made: str = ""
    
    def start_execution(self) -> None:
        """Mark the step as started."""
        self.start_datetime_utc = datetime.utcnow().isoformat() + "Z"
        self.status = StageStatus.RUNNING
        logger.info(f"Started execution of step {self.step_id}: {self.process_description}")
    
    def complete_execution(self, message: str = "", outputs: Optional[List[str]] = None) -> None:
        """Mark the step as completed."""
        self.end_datetime_utc = datetime.utcnow().isoformat() + "Z"
        self.status = StageStatus.COMPLETED
        self.message = message
        if outputs:
            self.outputs_generated.extend(outputs)
        
        # Calculate duration
        if self.start_datetime_utc:
            start_time = datetime.fromisoformat(self.start_datetime_utc.replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(self.end_datetime_utc.replace('Z', '+00:00'))
            self.duration_seconds = (end_time - start_time).total_seconds()
        
        duration = self.duration_seconds if self.duration_seconds is not None else 0.0
        logger.info(f"Completed step {self.step_id} in {duration:.2f}s")
    
    def fail_execution(self, error_details: Dict[str, Any], message: str = "") -> None:
        """Mark the step as failed."""
        self.end_datetime_utc = datetime.utcnow().isoformat() + "Z"
        self.status = StageStatus.FAILED
        self.error_details = error_details
        self.message = message or f"Step failed: {error_details.get('message', 'Unknown error')}"
        logger.error(f"Step {self.step_id} failed: {self.message}")


@dataclass
class CurrentStage:
    """Represents the current execution stage."""
    name: str
    step_id: str
    agent_name: str
    started_at: str
    completed_at: Optional[str] = None
    status: StageStatus = StageStatus.RUNNING
    message: str = ""
    
    def complete(self, message: str = "") -> None:
        """Mark the current stage as completed."""
        self.completed_at = datetime.utcnow().isoformat() + "Z"
        self.status = StageStatus.COMPLETED
        self.message = message
        logger.info(f"Completed stage {self.name} ({self.step_id})")


@dataclass
class Objective:
    """Represents a workflow objective with SMART criteria."""
    name: str
    target: str
    description: str
    status: str = "IN_PROGRESS"
    completion_percentage: float = 0.0
    assigned_agent: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    
    def update_progress(self, percentage: float, status: Optional[str] = None) -> None:
        """Update objective progress."""
        self.completion_percentage = min(100.0, max(0.0, percentage))
        if status:
            self.status = status
        logger.info(f"Objective '{self.name}' progress: {self.completion_percentage:.1f}%")


@dataclass
class KPI:
    """Represents a Key Performance Indicator."""
    name: str
    target: str
    description: str
    status: str = "IN_PROGRESS"
    current_value: Optional[Union[float, str]] = None
    measurement_unit: str = ""
    baseline_value: Optional[Union[float, str]] = None
    
    def update_value(self, value: Union[float, str], status: Optional[str] = None) -> None:
        """Update KPI current value."""
        self.current_value = value
        if status:
            self.status = status
        logger.info(f"KPI '{self.name}' updated to: {self.current_value} {self.measurement_unit}")


class WorkflowState:
    """
    Comprehensive workflow state management class.
    
    This class manages the complete state of a workflow including execution history,
    current status, objectives, KPIs, and provides methods for state transitions.
    """
    
    def __init__(self, workflow_request_id: Optional[str] = None, user_query: str = ""):
        """
        Initialize workflow state.
        
        Args:
            workflow_request_id: Unique identifier for the workflow
            user_query: Original user prompt that initiated the workflow
        """
        self.workflow_request_id = workflow_request_id or str(uuid4())
        self.creation_datetime_utc = datetime.utcnow().isoformat() + "Z"
        self.last_update_datetime_utc = self.creation_datetime_utc
        self.user_query = user_query
        self.status = WorkflowStatus.INITIALIZED
        self.last_failed_step_id = ""
        self.workflow_output_folder_path = ""
        self.general_workflow_log_path = ""
        self.keywords: List[str] = []
        
        # Execution tracking
        self.current_stage: Optional[CurrentStage] = None
        self.execution_history: List[ExecutionStep] = []
        
        # Project definition
        self.project_name = ""
        self.objectives: List[Objective] = []
        self.kpis: List[KPI] = []
        self.baseline = ""
        
        # Quality and success tracking
        self.successes: List[str] = []
        self.failures: List[str] = []
        self.qc_against_plan_status = "PENDING"
        self.qc_against_plan_notes = ""
        self.project_grade: Optional[float] = None
        
        logger.info(f"Initialized workflow state for request {self.workflow_request_id}")
    
    def update_status(self, new_status: WorkflowStatus, message: str = "") -> None:
        """
        Update workflow status with timestamp.
        
        Args:
            new_status: New workflow status
            message: Optional message describing the status change
        """
        old_status = self.status
        self.status = new_status
        self.last_update_datetime_utc = datetime.utcnow().isoformat() + "Z"
        
        logger.info(f"Workflow {self.workflow_request_id} status changed: {old_status.value} -> {new_status.value}")
        if message:
            logger.info(f"Status change message: {message}")
    
    def add_execution_step(self, step: ExecutionStep) -> None:
        """Add a new execution step to the history."""
        self.execution_history.append(step)
        self.last_update_datetime_utc = datetime.utcnow().isoformat() + "Z"
        logger.info(f"Added execution step {step.step_id} to workflow {self.workflow_request_id}")
    
    def get_current_step(self) -> Optional[ExecutionStep]:
        """Get the currently running step, if any."""
        for step in reversed(self.execution_history):
            if step.status == StageStatus.RUNNING:
                return step
        return None
    
    def get_failed_steps(self) -> List[ExecutionStep]:
        """Get all failed steps."""
        return [step for step in self.execution_history if step.status == StageStatus.FAILED]
    
    def get_completed_steps(self) -> List[ExecutionStep]:
        """Get all completed steps."""
        return [step for step in self.execution_history if step.status == StageStatus.COMPLETED]
    
    def calculate_progress_percentage(self) -> float:
        """Calculate overall workflow progress percentage."""
        if not self.execution_history:
            return 0.0
        
        completed = len(self.get_completed_steps())
        total = len(self.execution_history)
        return (completed / total) * 100.0
    
    def add_objective(self, name: str, target: str, description: str) -> Objective:
        """Add a new objective to the workflow."""
        objective = Objective(name=name, target=target, description=description)
        self.objectives.append(objective)
        self.last_update_datetime_utc = datetime.utcnow().isoformat() + "Z"
        logger.info(f"Added objective '{name}' to workflow {self.workflow_request_id}")
        return objective
    
    def add_kpi(self, name: str, target: str, description: str, measurement_unit: str = "") -> KPI:
        """Add a new KPI to the workflow."""
        kpi = KPI(name=name, target=target, description=description, measurement_unit=measurement_unit)
        self.kpis.append(kpi)
        self.last_update_datetime_utc = datetime.utcnow().isoformat() + "Z"
        logger.info(f"Added KPI '{name}' to workflow {self.workflow_request_id}")
        return kpi
    
    def record_success(self, success_description: str) -> None:
        """Record a success event."""
        self.successes.append(success_description)
        self.last_update_datetime_utc = datetime.utcnow().isoformat() + "Z"
        logger.info(f"Recorded success: {success_description}")
    
    def record_failure(self, failure_description: str) -> None:
        """Record a failure event."""
        self.failures.append(failure_description)
        self.last_update_datetime_utc = datetime.utcnow().isoformat() + "Z"
        logger.warning(f"Recorded failure: {failure_description}")
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of execution statistics."""
        total_steps = len(self.execution_history)
        completed_steps = len(self.get_completed_steps())
        failed_steps = len(self.get_failed_steps())
        running_steps = len([s for s in self.execution_history if s.status == StageStatus.RUNNING])
        
        total_duration = sum(
            step.duration_seconds for step in self.execution_history 
            if step.duration_seconds is not None
        )
        
        return {
            'total_steps': total_steps,
            'completed_steps': completed_steps,
            'failed_steps': failed_steps,
            'running_steps': running_steps,
            'progress_percentage': self.calculate_progress_percentage(),
            'total_duration_seconds': total_duration,
            'current_status': self.status.value,
            'objectives_count': len(self.objectives),
            'kpis_count': len(self.kpis),
            'successes_count': len(self.successes),
            'failures_count': len(self.failures)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow state to dictionary representation."""
        return {
            'workflow_request_id': self.workflow_request_id,
            'creation_datetime_utc': self.creation_datetime_utc,
            'last_update_datetime_utc': self.last_update_datetime_utc,
            'user_query': self.user_query,
            'status': self.status.value,
            'last_failed_step_id': self.last_failed_step_id,
            'workflow_output_folder_path': self.workflow_output_folder_path,
            'general_workflow_log_path': self.general_workflow_log_path,
            'keywords': self.keywords,
            'project_name': self.project_name,
            'objectives': [obj.__dict__ for obj in self.objectives],
            'kpis': [kpi.__dict__ for kpi in self.kpis],
            'baseline': self.baseline,
            'current_stage': self.current_stage.__dict__ if self.current_stage else None,
            'execution_history': [step.__dict__ for step in self.execution_history],
            'successes': self.successes,
            'failures': self.failures,
            'qc_against_plan_status': self.qc_against_plan_status,
            'qc_against_plan_notes': self.qc_against_plan_notes,
            'project_grade': self.project_grade,
            'execution_summary': self.get_execution_summary()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowState':
        """Create WorkflowState instance from dictionary."""
        instance = cls(
            workflow_request_id=data.get('workflow_request_id'),
            user_query=data.get('user_query', '')
        )
        
        # Restore basic attributes
        instance.creation_datetime_utc = data.get('creation_datetime_utc', instance.creation_datetime_utc)
        instance.last_update_datetime_utc = data.get('last_update_datetime_utc', instance.last_update_datetime_utc)
        instance.status = WorkflowStatus(data.get('status', 'INITIALIZED'))
        instance.last_failed_step_id = data.get('last_failed_step_id', '')
        instance.workflow_output_folder_path = data.get('workflow_output_folder_path', '')
        instance.general_workflow_log_path = data.get('general_workflow_log_path', '')
        instance.keywords = data.get('keywords', [])
        instance.project_name = data.get('project_name', '')
        instance.baseline = data.get('baseline', '')
        instance.successes = data.get('successes', [])
        instance.failures = data.get('failures', [])
        instance.qc_against_plan_status = data.get('qc_against_plan_status', 'PENDING')
        instance.qc_against_plan_notes = data.get('qc_against_plan_notes', '')
        instance.project_grade = data.get('project_grade')
        
        # Restore objectives
        for obj_data in data.get('objectives', []):
            obj = Objective(**obj_data)
            instance.objectives.append(obj)
        
        # Restore KPIs
        for kpi_data in data.get('kpis', []):
            kpi = KPI(**kpi_data)
            instance.kpis.append(kpi)
        
        # Restore execution history
        for step_data in data.get('execution_history', []):
            step_data['status'] = StageStatus(step_data.get('status', 'PENDING'))
            step = ExecutionStep(**step_data)
            instance.execution_history.append(step)
        
        # Restore current stage
        current_stage_data = data.get('current_stage')
        if current_stage_data:
            current_stage_data['status'] = StageStatus(current_stage_data.get('status', 'RUNNING'))
            instance.current_stage = CurrentStage(**current_stage_data)
        
        return instance
