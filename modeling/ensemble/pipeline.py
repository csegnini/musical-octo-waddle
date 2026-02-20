"""
AgentEnsemblePipeline: Highly Adaptable, Agent-Driven, Workflow-Orchestrated, and Documented Ensemble Pipeline

This pipeline enables modular or full execution of ensemble tasks by an agent, using workflow_orchestrator and workflows modules. The agent can assemble the workflow from config/capabilities and use the pipelines module in combination with workflow modules.
"""

from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
import pandas as pd
from . import VotingEnsembleModel, BaggingEnsembleModel

class AgentEnsemblePipeline:
    """
    Highly adaptable ensemble pipeline for agent-driven, workflow-orchestrated, and documented execution.
    """
    def __init__(self,
                 ensemble_type: str = 'voting',
                 base_estimators: Optional[List[Any]] = None,
                 ensemble_params: Optional[Dict[str, Any]] = None,
                 workflow_manager=None,
                 doc_engine=None,
                 agent_config: Optional[Dict[str, Any]] = None):
        self.ensemble_type = ensemble_type
        self.base_estimators = base_estimators or []
        self.ensemble_params = ensemble_params or {}
        self.workflow_manager = workflow_manager
        self.doc_engine = doc_engine
        self.agent_config = agent_config or {}
        self.model = self._create_model()
        self.is_fitted = False

    def _create_model(self):
        if self.ensemble_type == 'voting':
            return VotingEnsembleModel(base_estimators=self.base_estimators, **self.ensemble_params)
        elif self.ensemble_type == 'bagging':
            return BaggingEnsembleModel(base_estimator=self.base_estimators[0] if self.base_estimators else None, **self.ensemble_params)
        else:
            raise ValueError(f"Unknown ensemble_type: {self.ensemble_type}")

    def run_workflow(self, X, y, validation_data=None) -> 'AgentEnsemblePipeline':
        """
        Agent-driven, orchestrator-based, and documented execution of ensemble pipeline.
        Uses orchestrator Workflow, Task, WorkflowEngine, and always integrates WorkflowManager/DocumentationEngine.
        Legacy direct logic is deprecated but preserved for reference.
        Args:
            X: Input features
            y: Target variable
            validation_data: Optional validation data
        Returns:
            self: Fitted pipeline
        """
        # Always use WorkflowManager and DocumentationEngine
        if self.workflow_manager is None:
            from src.workflows.workflow_manager import WorkflowManager
            self.workflow_manager = WorkflowManager(user_query="ensemble workflow")
        if self.doc_engine is None:
            from src.workflows.documentation_engine import DocumentationEngine
            self.doc_engine = DocumentationEngine(self.workflow_manager)

        # Define modular tasks for orchestrator
        def task_fit_ensemble_model(X, y, validation_data=None):
            step_id = "ensemble_pipeline_fit"
            self.workflow_manager.add_execution_step(
                step_id=step_id,
                macro_process_name="Ensemble",
                process_description="Fit ensemble model",
                responsible_agent_or_tool="AgentEnsemblePipeline",
                inputs_used=[str(np.shape(X)), str(np.shape(y))]
            )
            self.workflow_manager.start_execution_step(step_id)
            self.model.fit(X, y)
            self.is_fitted = True
            self.workflow_manager.complete_execution_step(
                step_id,
                message="Completed ensemble model fit",
                outputs=[str(np.shape(X)), str(np.shape(y))]
            )
            self.doc_engine.generate_html_report()
            return "fit_complete"

        # Assemble orchestrator workflow
        from src.workflow_orchestrator.core import Workflow, Task, WorkflowEngine
        from src.workflow_orchestrator.utils import WorkflowBuilder

        workflow = WorkflowBuilder(
            name="EnsemblePipelineWorkflow",
            description="Agent-driven modular ensemble pipeline workflow"
        ).add_task(
            name="FitEnsembleModel",
            function=lambda X=X, y=y, validation_data=validation_data: task_fit_ensemble_model(X, y, validation_data),
            parameters={"X": X, "y": y, "validation_data": validation_data}
        ).build()

        # Register and run workflow
        engine = WorkflowEngine()
        engine.register_workflow(workflow)
        result = engine.run(workflow)

        # Document workflow execution (summary step)
        self.workflow_manager.add_execution_step(
            step_id="ensemble_workflow_completed",
            macro_process_name="Ensemble",
            process_description="Ensemble workflow completed",
            responsible_agent_or_tool="AgentEnsemblePipeline",
            inputs_used=[str(np.shape(X)), str(np.shape(y))]
        )
        self.workflow_manager.start_execution_step("ensemble_workflow_completed")
        self.workflow_manager.complete_execution_step(
            "ensemble_workflow_completed",
            message="Ensemble workflow completed",
            outputs=[str(np.shape(X)), str(np.shape(y))]
        )
        self.doc_engine.generate_html_report()

        # DEPRECATED: Legacy direct logic (preserved for reference)
        # self.model.fit(X, y)
        # self.is_fitted = True

        return self

    def predict(self, X):
        """Make predictions using the fitted ensemble model."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before making predictions")
        return self.model.predict(X)

    def predict_proba(self, X):
        """Get prediction probabilities (if supported)."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before making predictions")
        return self.model.predict_proba(X)

    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        summary = f"Ensemble type: {self.ensemble_type}\nFitted: {self.is_fitted}"
        if hasattr(self.model, 'ensemble_model'):
            summary += f"\nBase estimators: {getattr(self.model, 'base_estimators', [])}"
        return summary

    def generate_report(self) -> str:
        """Generate a comprehensive ensemble report (if supported)."""
        summary = self.get_model_summary()
        return f"Ensemble Pipeline Report\n{'='*40}\n{summary}"
