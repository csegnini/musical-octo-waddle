"""
Workflow Updater Module

This module provides a simple, thread-safe interface for external methods to update
workflow state, add data references, and manage work orders without direct access
to the underlying workflow management components.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from threading import Lock
from datetime import datetime

from .workflow_manager import WorkflowManager
from .workflow_state import WorkflowStatus, StageStatus
from .data_inventory import DataType
from .agent_coordinator import WorkOrderStatus

logger = logging.getLogger(__name__)


class WorkflowUpdater:
    """
    Thread-safe interface for updating workflow components.
    
    This class provides a simplified API for external methods to interact with
    the workflow management system without needing direct access to internal
    components.
    """
    
    def __init__(self, workflow_manager: WorkflowManager):
        """
        Initialize workflow updater.
        
        Args:
            workflow_manager: The workflow manager instance to update
        """
        self.workflow_manager = workflow_manager
        self._lock = Lock()
        
        logger.info(f"Initialized workflow updater for workflow {workflow_manager.workflow_id}")
    
    def update_status(self, status: Union[WorkflowStatus, str], message: str = "") -> bool:
        """
        Update the overall workflow status.
        
        Args:
            status: New workflow status (enum or string)
            message: Optional status message
            
        Returns:
            True if update successful
        """
        try:
            with self._lock:
                if isinstance(status, str):
                    status = WorkflowStatus(status)
                
                self.workflow_manager.update_workflow_status(status, message)
                logger.info(f"Updated workflow status to {status.value}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating workflow status: {e}")
            return False
    
    def add_step(
        self,
        step_id: str,
        process_name: str,
        description: str,
        responsible_agent: str,
        inputs: Optional[List[str]] = None
    ) -> bool:
        """
        Add a new execution step to the workflow.
        
        Args:
            step_id: Unique step identifier
            process_name: Name of the process/macro
            description: Step description
            responsible_agent: Agent or tool responsible
            inputs: List of input references
            
        Returns:
            True if step added successfully
        """
        try:
            with self._lock:
                self.workflow_manager.add_execution_step(
                    step_id=step_id,
                    macro_process_name=process_name,
                    process_description=description,
                    responsible_agent_or_tool=responsible_agent,
                    inputs_used=inputs
                )
                logger.info(f"Added execution step {step_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error adding execution step: {e}")
            return False
    
    def start_step(self, step_id: str) -> bool:
        """
        Mark a step as started.
        
        Args:
            step_id: Step identifier
            
        Returns:
            True if step started successfully
        """
        try:
            with self._lock:
                success = self.workflow_manager.start_execution_step(step_id)
                if success:
                    logger.info(f"Started execution step {step_id}")
                return success
                
        except Exception as e:
            logger.error(f"Error starting step {step_id}: {e}")
            return False
    
    def complete_step(
        self, 
        step_id: str, 
        message: str = "", 
        outputs: Optional[List[str]] = None
    ) -> bool:
        """
        Mark a step as completed.
        
        Args:
            step_id: Step identifier
            message: Completion message
            outputs: List of output references
            
        Returns:
            True if step completed successfully
        """
        try:
            with self._lock:
                success = self.workflow_manager.complete_execution_step(step_id, message, outputs)
                if success:
                    logger.info(f"Completed execution step {step_id}")
                return success
                
        except Exception as e:
            logger.error(f"Error completing step {step_id}: {e}")
            return False
    
    def fail_step(
        self, 
        step_id: str, 
        error_message: str, 
        error_type: str = "GeneralError",
        traceback: str = ""
    ) -> bool:
        """
        Mark a step as failed.
        
        Args:
            step_id: Step identifier
            error_message: Error description
            error_type: Type of error
            traceback: Error traceback
            
        Returns:
            True if step failure recorded successfully
        """
        try:
            with self._lock:
                error_details = {
                    'error_type': error_type,
                    'message': error_message,
                    'traceback': traceback,
                    'timestamp': datetime.utcnow().isoformat() + "Z"
                }
                
                success = self.workflow_manager.fail_execution_step(step_id, error_details, error_message)
                if success:
                    logger.warning(f"Failed execution step {step_id}: {error_message}")
                return success
                
        except Exception as e:
            logger.error(f"Error failing step {step_id}: {e}")
            return False
    
    def add_data_file(
        self,
        file_path: str,
        data_type: Union[DataType, str],
        name: str = "",
        description: str = "",
        generated_by_step: str = "",
        record_count: int = 0,
        **metadata
    ) -> Optional[str]:
        """
        Add a data file to the inventory.
        
        Args:
            file_path: Path to the data file
            data_type: Type of data (enum or string)
            name: File name (auto-detected if not provided)
            description: Description of the data
            generated_by_step: Step that generated this data
            record_count: Number of records in the data
            **metadata: Additional metadata
            
        Returns:
            Data ID if successful, None otherwise
        """
        try:
            with self._lock:
                if isinstance(data_type, str):
                    data_type = DataType(data_type)
                
                data_ref = self.workflow_manager.add_data_file(
                    data_type=data_type,
                    file_path=file_path,
                    name=name,
                    description=description,
                    generated_by_step_id=generated_by_step,
                    record_count=record_count,
                    **metadata
                )
                
                logger.info(f"Added data file {file_path} with ID {data_ref.data_id}")
                return data_ref.data_id
                
        except Exception as e:
            logger.error(f"Error adding data file {file_path}: {e}")
            return None
    
    def create_work_order(
        self,
        instructions: str,
        process_name: str = "",
        input_data_ids: Optional[List[str]] = None,
        expected_outputs: Optional[List[str]] = None,
        tool_library: Optional[str] = None,
        tool_methods: Optional[List[str]] = None,
        priority: int = 0
    ) -> Optional[str]:
        """
        Create a new work order.
        
        Args:
            instructions: Detailed instructions for the work
            process_name: Name of the process/macro
            input_data_ids: List of input data IDs
            expected_outputs: List of expected output paths
            tool_library: Library to use
            tool_methods: Methods to use
            priority: Priority level
            
        Returns:
            Work order ID if successful, None otherwise
        """
        try:
            with self._lock:
                work_order = self.workflow_manager.create_work_order(
                    instructions=instructions,
                    macro_process=process_name,
                    input_data_ids=input_data_ids,
                    expected_outputs=expected_outputs,
                    tool_library=tool_library,
                    tool_methods=tool_methods,
                    priority=priority
                )
                
                logger.info(f"Created work order {work_order.order_id}")
                return work_order.order_id
                
        except Exception as e:
            logger.error(f"Error creating work order: {e}")
            return None
    
    def assign_work_order(self, order_id: str, agent_id: str) -> bool:
        """
        Assign a work order to an agent.
        
        Args:
            order_id: Work order ID
            agent_id: Agent ID
            
        Returns:
            True if assignment successful
        """
        try:
            with self._lock:
                success = self.workflow_manager.assign_work_order(order_id, agent_id)
                if success:
                    logger.info(f"Assigned work order {order_id} to agent {agent_id}")
                return success
                
        except Exception as e:
            logger.error(f"Error assigning work order {order_id}: {e}")
            return False
    
    def complete_work_order(
        self, 
        order_id: str, 
        result_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Complete a work order.
        
        Args:
            order_id: Work order ID
            result_data: Optional result data
            
        Returns:
            True if completion successful
        """
        try:
            with self._lock:
                success = self.workflow_manager.complete_work_order(order_id, result_data)
                if success:
                    logger.info(f"Completed work order {order_id}")
                return success
                
        except Exception as e:
            logger.error(f"Error completing work order {order_id}: {e}")
            return False
    
    def fail_work_order(
        self, 
        order_id: str, 
        error_message: str, 
        error_type: str = "ExecutionError"
    ) -> bool:
        """
        Fail a work order.
        
        Args:
            order_id: Work order ID
            error_message: Error description
            error_type: Type of error
            
        Returns:
            True if failure recorded successfully
        """
        try:
            with self._lock:
                error_details = {
                    'error_type': error_type,
                    'message': error_message,
                    'timestamp': datetime.utcnow().isoformat() + "Z"
                }
                
                success = self.workflow_manager.fail_work_order(order_id, error_details)
                if success:
                    logger.warning(f"Failed work order {order_id}: {error_message}")
                return success
                
        except Exception as e:
            logger.error(f"Error failing work order {order_id}: {e}")
            return False
    
    def add_objective(self, name: str, target: str, description: str) -> bool:
        """
        Add an objective to the workflow.
        
        Args:
            name: Objective name
            target: Target/goal
            description: Detailed description
            
        Returns:
            True if objective added successfully
        """
        try:
            with self._lock:
                self.workflow_manager.add_objective(name, target, description)
                logger.info(f"Added objective: {name}")
                return True
                
        except Exception as e:
            logger.error(f"Error adding objective {name}: {e}")
            return False
    
    def add_kpi(
        self, 
        name: str, 
        target: str, 
        description: str, 
        unit: str = ""
    ) -> bool:
        """
        Add a KPI to the workflow.
        
        Args:
            name: KPI name
            target: Target value
            description: Description
            unit: Measurement unit
            
        Returns:
            True if KPI added successfully
        """
        try:
            with self._lock:
                self.workflow_manager.add_kpi(name, target, description, unit)
                logger.info(f"Added KPI: {name}")
                return True
                
        except Exception as e:
            logger.error(f"Error adding KPI {name}: {e}")
            return False
    
    def record_success(self, description: str) -> bool:
        """
        Record a success event.
        
        Args:
            description: Success description
            
        Returns:
            True if success recorded
        """
        try:
            with self._lock:
                self.workflow_manager.record_success(description)
                logger.info(f"Recorded success: {description}")
                return True
                
        except Exception as e:
            logger.error(f"Error recording success: {e}")
            return False
    
    def record_failure(self, description: str) -> bool:
        """
        Record a failure event.
        
        Args:
            description: Failure description
            
        Returns:
            True if failure recorded
        """
        try:
            with self._lock:
                self.workflow_manager.record_failure(description)
                logger.warning(f"Recorded failure: {description}")
                return True
                
        except Exception as e:
            logger.error(f"Error recording failure: {e}")
            return False
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a workflow summary.
        
        Returns:
            Workflow summary dictionary
        """
        try:
            with self._lock:
                return self.workflow_manager.get_workflow_summary()
                
        except Exception as e:
            logger.error(f"Error getting workflow summary: {e}")
            return {}
    
    def get_current_status(self) -> str:
        """
        Get the current workflow status.
        
        Returns:
            Current workflow status as string
        """
        try:
            with self._lock:
                return self.workflow_manager.workflow_state.status.value
                
        except Exception as e:
            logger.error(f"Error getting current status: {e}")
            return "UNKNOWN"
    
    def export_documentation(self, file_path: Optional[str] = None) -> Optional[str]:
        """
        Export workflow documentation.
        
        Args:
            file_path: Optional export path
            
        Returns:
            Path to exported file if successful, None otherwise
        """
        try:
            with self._lock:
                return self.workflow_manager.export_workflow_package(file_path)
                
        except Exception as e:
            logger.error(f"Error exporting documentation: {e}")
            return None
    
    def auto_assign_orders(self) -> int:
        """
        Automatically assign pending work orders to available agents.
        
        Returns:
            Number of orders assigned
        """
        try:
            with self._lock:
                return self.workflow_manager.auto_assign_work_orders()
                
        except Exception as e:
            logger.error(f"Error auto-assigning work orders: {e}")
            return 0


# Convenience functions for easy access
def create_workflow_updater(
    workflow_id: Optional[str] = None, 
    user_query: str = "",
    output_folder: Optional[str] = None
) -> WorkflowUpdater:
    """
    Create a new workflow updater with a fresh workflow manager.
    
    Args:
        workflow_id: Unique workflow identifier
        user_query: Original user query
        output_folder: Output folder path
        
    Returns:
        WorkflowUpdater instance
    """
    workflow_manager = WorkflowManager(workflow_id, user_query, output_folder)
    return WorkflowUpdater(workflow_manager)


def get_workflow_updater(workflow_manager: WorkflowManager) -> WorkflowUpdater:
    """
    Get a workflow updater for an existing workflow manager.
    
    Args:
        workflow_manager: Existing workflow manager
        
    Returns:
        WorkflowUpdater instance
    """
    return WorkflowUpdater(workflow_manager)
