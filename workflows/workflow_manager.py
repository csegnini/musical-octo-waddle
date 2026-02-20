"""
Workflow Manager Module

This module provides the main workflow management interface that orchestrates
workflow state, data inventory, agent coordination, and documentation generation.
"""

import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from uuid import uuid4

from .workflow_state import WorkflowState, WorkflowStatus, ExecutionStep, StageStatus
from .data_inventory import DataInventoryManager, DataReference, DataType
from .agent_coordinator import AgentCoordinator, WorkOrder, ToolConfig

logger = logging.getLogger(__name__)


class WorkflowManager:
    def run_workflow(
            self,
            workflow_name: str,
            tasks: list,
            description: str = "",
            orchestrator_config: dict = None,
            executor_type: str = "parallel"
        ):
            """
            Unified entry point to build and execute an orchestrator-based workflow.

            Args:
                workflow_name: Name of the workflow
                tasks: List of orchestrator Task objects
                description: Workflow description
                orchestrator_config: Optional config for WorkflowEngine
                executor_type: 'parallel' or 'sequential'
            Returns:
                WorkflowResult object
            """
            from src.workflow_orchestrator.core import Workflow, WorkflowEngine
            # Build orchestrator workflow
            workflow = Workflow(
                name=workflow_name,
                workflow_id=self.workflow_id,
                description=description,
                tasks=tasks,
                metadata={"user_query": self.user_query}
            )
            engine = WorkflowEngine(config=orchestrator_config)
            engine.register_workflow(workflow)
            result = engine.run(workflow, executor_type=executor_type)
            # Optionally update documentation, state, etc.
            self.update_workflow_status(result.status)
            return result

            """
            Comprehensive workflow management system.
    
            This class provides a unified interface for managing multi-agent workflows,
            including state management, data inventory, agent coordination, and documentation.
            """

    def __init__(
        self, 
        workflow_request_id: Optional[str] = None, 
        user_query: str = "",
        output_folder: Optional[str] = None
    ):
        """
        Initialize workflow manager.
        
        Args:
            workflow_request_id: Unique workflow identifier
            user_query: Original user prompt that initiated the workflow
            output_folder: Base folder for workflow outputs
        """
        self.workflow_id = workflow_request_id or str(uuid4())
        self.user_query = user_query
        
        # Initialize core components
        self.workflow_state = WorkflowState(self.workflow_id, user_query)
        self.data_inventory = DataInventoryManager(self.workflow_id)
        self.agent_coordinator = AgentCoordinator(self.workflow_id)
        
        # Setup output folder structure
        self.output_folder = output_folder or f"workflows/workflow_{self.workflow_id[:8]}"
        self.setup_folder_structure()
        
        # Workflow documentation file
        self.documentation_file = os.path.join(self.output_folder, "workflow_documentation.json")
        
        logger.info(f"Initialized workflow manager for workflow {self.workflow_id}")
    
    def setup_folder_structure(self) -> None:
        """Create the folder structure for the workflow."""
        try:
            base_path = Path(self.output_folder)
            
            # Create main directories
            directories = [
                "data/raw",
                "data/processed", 
                "data/intermediate",
                "data/final",
                "logs",
                "reports",
                "models",
                "configs",
                "schemas"
            ]
            
            for directory in directories:
                (base_path / directory).mkdir(parents=True, exist_ok=True)
            
            # Update workflow state with folder paths
            self.workflow_state.workflow_output_folder_path = str(base_path.absolute())
            self.workflow_state.general_workflow_log_path = str((base_path / "logs" / "workflow.log").absolute())
            
            logger.info(f"Created folder structure at {self.output_folder}")
            
        except Exception as e:
            logger.error(f"Error creating folder structure: {e}")
    
    def update_workflow_status(self, status: WorkflowStatus, message: str = "") -> None:
        """
        Update the overall workflow status.
        
        Args:
            status: New workflow status
            message: Optional status message
        """
        self.workflow_state.update_status(status, message)
        self.save_documentation()
        logger.info(f"Workflow status updated to: {status.value}")
    
    def add_execution_step(
        self,
        step_id: str,
        macro_process_name: str,
        process_description: str,
        responsible_agent_or_tool: str,
        inputs_used: Optional[List[str]] = None
    ) -> ExecutionStep:
        """
        Add a new execution step to the workflow.
        
        Args:
            step_id: Unique step identifier
            macro_process_name: Name of the macro process
            process_description: Description of the process
            responsible_agent_or_tool: Name of responsible agent or tool
            inputs_used: List of input references
            
        Returns:
            Created ExecutionStep object
        """
        step = ExecutionStep(
            step_id=step_id,
            macro_process_name=macro_process_name,
            process_description=process_description,
            responsible_agent_or_tool=responsible_agent_or_tool,
            inputs_used=inputs_used or []
        )
        
        self.workflow_state.add_execution_step(step)
        self.save_documentation()
        
        return step
    
    def start_execution_step(self, step_id: str) -> bool:
        """
        Start execution of a specific step.
        
        Args:
            step_id: Step identifier
            
        Returns:
            True if step started successfully
        """
        # Find the step in execution history
        step = None
        for s in self.workflow_state.execution_history:
            if s.step_id == step_id:
                step = s
                break
        
        if not step:
            logger.error(f"Execution step {step_id} not found")
            return False
        
        step.start_execution()
        self.save_documentation()
        return True
    
    def complete_execution_step(
        self, 
        step_id: str, 
        message: str = "", 
        outputs: Optional[List[str]] = None
    ) -> bool:
        """
        Complete execution of a specific step.
        
        Args:
            step_id: Step identifier
            message: Completion message
            outputs: List of output references
            
        Returns:
            True if step completed successfully
        """
        # Find the step in execution history
        step = None
        for s in self.workflow_state.execution_history:
            if s.step_id == step_id:
                step = s
                break
        
        if not step:
            logger.error(f"Execution step {step_id} not found")
            return False
        
        step.complete_execution(message, outputs)
        self.save_documentation()
        return True
    
    def fail_execution_step(
        self, 
        step_id: str, 
        error_details: Dict[str, Any], 
        message: str = ""
    ) -> bool:
        """
        Fail execution of a specific step.
        
        Args:
            step_id: Step identifier
            error_details: Error details dictionary
            message: Error message
            
        Returns:
            True if step failed successfully
        """
        # Find the step in execution history
        step = None
        for s in self.workflow_state.execution_history:
            if s.step_id == step_id:
                step = s
                break
        
        if not step:
            logger.error(f"Execution step {step_id} not found")
            return False
        
        step.fail_execution(error_details, message)
        self.workflow_state.last_failed_step_id = step_id
        self.save_documentation()
        return True
    
    def add_data_file(
        self,
        data_type: DataType,
        file_path: str,
        name: str = "",
        description: str = "",
        generated_by_step_id: str = "",
        **kwargs
    ) -> DataReference:
        """
        Add a data file to the inventory.
        
        Args:
            data_type: Type of data
            file_path: Path to the data file
            name: Name of the data file
            description: Description of the data
            generated_by_step_id: ID of the step that generated this data
            **kwargs: Additional parameters
            
        Returns:
            Created DataReference object
        """
        if not name:
            name = os.path.basename(file_path)
        
        data_ref = self.data_inventory.create_and_add_data_reference(
            data_type=data_type,
            name=name,
            path=file_path,
            description=description,
            generated_by_step_id=generated_by_step_id,
            **kwargs
        )
        
        self.save_documentation()
        return data_ref
    
    def create_work_order(
        self,
        instructions: str,
        macro_process: str = "",
        input_data_ids: Optional[List[str]] = None,
        expected_outputs: Optional[List[str]] = None,
        tool_library: Optional[str] = None,
        tool_methods: Optional[List[str]] = None,
        priority: int = 0
    ) -> WorkOrder:
        """
        Create a work order for agent execution.
        
        Args:
            instructions: Detailed instructions
            macro_process: Name of the macro process
            input_data_ids: List of input data IDs
            expected_outputs: List of expected output file paths
            tool_library: Library to use for the task
            tool_methods: Methods to use from the library
            priority: Priority level
            
        Returns:
            Created WorkOrder object
        """
        # Create tool config if specified
        tool_config = None
        if tool_library:
            tool_config = ToolConfig(
                library=tool_library,
                methods=tool_methods or []
            )
        
        work_order = self.agent_coordinator.create_work_order(
            instructions=instructions,
            macro_process=macro_process,
            input_references=input_data_ids or [],
            output_references_expected=expected_outputs or [],
            tool_config=tool_config,
            priority=priority
        )
        
        self.save_documentation()
        return work_order
    
    def register_agent(self, agent_id: str, name: str, capabilities: Optional[List[str]] = None):
        """
        Register an agent in the coordination system.
        
        Args:
            agent_id: Unique agent identifier
            name: Human-readable agent name
            capabilities: List of agent capabilities
        """
        agent = self.agent_coordinator.register_agent(agent_id, name, capabilities)
        self.save_documentation()
        return agent
    
    def assign_work_order(self, order_id: str, agent_id: str) -> bool:
        """
        Assign a work order to an agent.
        
        Args:
            order_id: Work order ID
            agent_id: Agent ID
            
        Returns:
            True if assignment successful
        """
        success = self.agent_coordinator.assign_work_order(order_id, agent_id)
        if success:
            self.save_documentation()
        return success
    
    def auto_assign_work_orders(self) -> int:
        """
        Automatically assign pending work orders to available agents.
        
        Returns:
            Number of work orders assigned
        """
        assigned_count = self.agent_coordinator.auto_assign_work_orders()
        if assigned_count > 0:
            self.save_documentation()
        return assigned_count
    
    def complete_work_order(self, order_id: str, result_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Complete a work order.
        
        Args:
            order_id: Work order ID
            result_data: Optional result data
            
        Returns:
            True if completion successful
        """
        success = self.agent_coordinator.complete_work_order(order_id, result_data)
        if success:
            self.save_documentation()
        return success
    
    def fail_work_order(self, order_id: str, error_details: Dict[str, Any]) -> bool:
        """
        Fail a work order.
        
        Args:
            order_id: Work order ID
            error_details: Error details
            
        Returns:
            True if failure recorded successfully
        """
        success = self.agent_coordinator.fail_work_order(order_id, error_details)
        if success:
            self.save_documentation()
        return success
    
    def add_objective(self, name: str, target: str, description: str):
        """Add an objective to the workflow."""
        objective = self.workflow_state.add_objective(name, target, description)
        self.save_documentation()
        return objective
    
    def add_kpi(self, name: str, target: str, description: str, measurement_unit: str = ""):
        """Add a KPI to the workflow."""
        kpi = self.workflow_state.add_kpi(name, target, description, measurement_unit)
        self.save_documentation()
        return kpi
    
    def record_success(self, success_description: str) -> None:
        """Record a success event."""
        self.workflow_state.record_success(success_description)
        self.save_documentation()
    
    def record_failure(self, failure_description: str) -> None:
        """Record a failure event."""
        self.workflow_state.record_failure(failure_description)
        self.save_documentation()
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get a comprehensive workflow summary."""
        execution_summary = self.workflow_state.get_execution_summary()
        coordinator_stats = self.agent_coordinator.get_coordinator_statistics()
        inventory_stats = self.data_inventory.get_inventory_statistics()
        
        return {
            'workflow_id': self.workflow_id,
            'user_query': self.user_query,
            'current_status': self.workflow_state.status.value,
            'output_folder': self.output_folder,
            'execution_summary': execution_summary,
            'data_inventory_summary': inventory_stats,
            'agent_coordination_summary': coordinator_stats,
            'objectives_count': len(self.workflow_state.objectives),
            'kpis_count': len(self.workflow_state.kpis),
            'project_grade': self.workflow_state.project_grade
        }
    
    def generate_full_documentation(self) -> Dict[str, Any]:
        """
        Generate complete workflow documentation in the format matching workflow_example.json.
        
        Returns:
            Complete workflow documentation dictionary
        """
        # Get current data inventory
        data_inventory_export = self.data_inventory.export_inventory_to_dict()
        
        # Convert data inventory to the expected format
        data_inventory_formatted = {}
        for data_type_str, data_refs in data_inventory_export['data_inventory'].items():
            if data_refs:  # Only include if there are references
                # Take the first reference as an example (or could aggregate)
                first_ref = next(iter(data_refs.values()))
                data_inventory_formatted[data_type_str] = first_ref
        
        # Format work orders for the "work_orders_per_agent" section
        work_orders_formatted = {}
        for order_id, work_order in self.agent_coordinator.work_orders.items():
            order_dict = work_order.to_dict()
            # Create a simplified version for the macro process
            macro_name = work_order.macro_process or f"macro_{len(work_orders_formatted) + 1}"
            work_orders_formatted[macro_name] = {
                'order_id': order_dict['order_id'],
                'instructions': order_dict['instructions'],
                'input_references': order_dict['input_references'],
                'output_references_expected': order_dict['output_references_expected'],
                'assigned_tool_config': order_dict['assigned_tool_config'],
                'log_path': order_dict['log_path']
            }
        
        # Format execution history
        execution_history = []
        for step in self.workflow_state.execution_history:
            step_dict = step.__dict__.copy()
            step_dict['status'] = step.status.value
            execution_history.append(step_dict)
        
        # Build the complete documentation structure
        documentation = {
            'workflow_request': {
                'workflow_request_id': self.workflow_state.workflow_request_id,
                'creation_datetime_utc': self.workflow_state.creation_datetime_utc,
                'last_update_datetime_utc': self.workflow_state.last_update_datetime_utc,
                'user_query': self.workflow_state.user_query,
                'workflow_status': self.workflow_state.status.value,
                'last_failed_step_id': self.workflow_state.last_failed_step_id,
                'workflow_output_folder_path': self.workflow_state.workflow_output_folder_path,
                'general_workflow_log_path': self.workflow_state.general_workflow_log_path,
                'keywords': self.workflow_state.keywords
            },
            'workflow_references': {
                'related_knowledge_path': [],
                'info': {},
                'plan_reasoning': {
                    'weighted_successes': [],
                    'weighted_failures': [],
                    'success_patterns': [],
                    'failure_patterns': [],
                    'do': [],
                    'dont': []
                }
            },
            'workflow_definition': {
                'project_name': self.workflow_state.project_name,
                'keywords': self.workflow_state.keywords,
                'general_requirements': {
                    'critical_to_quality_metrics': [],
                    'needs': [],
                    'scope': '',
                    'objectives': [obj.__dict__ for obj in self.workflow_state.objectives],
                    'kpis': [kpi.__dict__ for kpi in self.workflow_state.kpis],
                    'baseline': self.workflow_state.baseline
                },
                'general_plan': {
                    'macro_process': '',
                    'agents_and_tools_selection': [],
                    'monitoring_controls_audits': [],
                    'high_level_sipoc': {
                        'suppliers': [],
                        'inputs': [],
                        'process': [],
                        'outputs': [],
                        'customers': []
                    }
                }
            },
            'work_orders_per_agent': work_orders_formatted,
            'data_inventory': data_inventory_formatted,
            'analysis_and_modeling_details': {
                'model_details': {
                    'model_type': '',
                    'model_name': '',
                    'model_path': '',
                    'training_metrics': {},
                    'validation_metrics': {},
                    'feature_importance_path': '',
                    'hyperparameters_used': {}
                }
            },
            'results_and_insights': {
                'predictions_path': '',
                'visualizations_paths': [],
                'insights_summary_path': '',
                'recommendations_summary_path': '',
                'final_report_path': '',
                'objectives': 'achieved objectives',
                'KPIs': 'kpis achievement',
                'successes': self.workflow_state.successes,
                'failures': self.workflow_state.failures,
                'qc_against_plan_status': self.workflow_state.qc_against_plan_status,
                'qc_against_plan_notes': self.workflow_state.qc_against_plan_notes,
                'project_grade': self.workflow_state.project_grade
            },
            'workflow_execution': {
                'current_stage': self.workflow_state.current_stage.__dict__ if self.workflow_state.current_stage else None,
                'history': execution_history
            }
        }
        
        return documentation
    
    def save_documentation(self) -> None:
        """Save the current workflow documentation to file."""
        try:
            documentation = self.generate_full_documentation()
            
            with open(self.documentation_file, 'w', encoding='utf-8') as f:
                json.dump(documentation, f, indent=2, ensure_ascii=False, default=str)
            
            logger.debug(f"Saved workflow documentation to {self.documentation_file}")
            
        except Exception as e:
            logger.error(f"Error saving workflow documentation: {e}")
    
    def load_documentation(self, file_path: Optional[str] = None) -> bool:
        """
        Load workflow documentation from file.
        
        Args:
            file_path: Path to documentation file (uses default if None)
            
        Returns:
            True if loaded successfully
        """
        load_path = file_path or self.documentation_file
        
        try:
            if not os.path.exists(load_path):
                logger.warning(f"Documentation file not found: {load_path}")
                return False
            
            with open(load_path, 'r', encoding='utf-8') as f:
                documentation = json.load(f)
            
            # Restore workflow state from documentation
            # This would involve parsing the documentation and restoring state
            # Implementation depends on specific requirements
            
            logger.info(f"Loaded workflow documentation from {load_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading workflow documentation: {e}")
            return False
    
    def export_workflow_package(self, export_path: Optional[str] = None) -> str:
        """
        Export complete workflow package including documentation and data.
        
        Args:
            export_path: Path for export (auto-generated if None)
            
        Returns:
            Path to exported package
        """
        if not export_path:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            export_path = f"workflow_export_{self.workflow_id[:8]}_{timestamp}.json"
        
        try:
            # Generate complete documentation
            documentation = self.generate_full_documentation()
            
            # Add summary information
            package = {
                'export_metadata': {
                    'export_timestamp': datetime.utcnow().isoformat() + "Z",
                    'workflow_id': self.workflow_id,
                    'export_version': '1.0',
                    'summary': self.get_workflow_summary()
                },
                'workflow_documentation': documentation,
                'data_inventory_full': self.data_inventory.export_inventory_to_dict(),
                'agent_coordination_full': self.agent_coordinator.export_to_dict()
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(package, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Exported workflow package to {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"Error exporting workflow package: {e}")
            raise
