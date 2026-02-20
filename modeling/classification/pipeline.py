"""
Classification Pipeline for Agent-Driven, Workflow-Orchestrated, and Documented Execution

This module provides a highly adaptable classification pipeline class that can be orchestrated by an agent using the workflow_orchestrator and documented using the workflows module. The pipeline can be assembled from config/capabilities and supports integration with the agent, WorkflowManager, DocumentationEngine, and other workflow modules.
"""

from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
import pandas as pd
from . import (
    LogisticRegressionModel, SVMModel, DecisionTreeModel, RandomForestModel,
    create_logistic_regression, create_svm, create_decision_tree, create_random_forest
)

class AgentClassificationPipeline:
    """
    Highly adaptable classification pipeline for agent-driven, workflow-orchestrated, and documented execution.
    """
    def __init__(self,
                 model_type: str = 'random_forest',
                 model_params: Optional[Dict[str, Any]] = None,
                 workflow_manager=None,
                 doc_engine=None,
                 agent_config: Optional[Dict[str, Any]] = None):
        self.model_type = model_type
        self.model_params = model_params or {}
        self.workflow_manager = workflow_manager
        self.doc_engine = doc_engine
        self.agent_config = agent_config or {}
        self.model = self._create_model()
        self.is_fitted = False

    def _create_model(self):
        if self.model_type == 'logistic_regression':
            return create_logistic_regression(**self.model_params)
        elif self.model_type == 'svm':
            return create_svm(**self.model_params)
        elif self.model_type == 'decision_tree':
            return create_decision_tree(**self.model_params)
        else:
            return create_random_forest(**self.model_params)

    def run_workflow(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'AgentClassificationPipeline':
        """
        Agent-driven, orchestrator-based, and documented execution of classification pipeline.
        Uses orchestrator Workflow, Task, WorkflowEngine, and always integrates WorkflowManager/DocumentationEngine.
        Legacy direct logic is deprecated but preserved for reference.
        Args:
            X: Input features
            y: Target variable
        Returns:
            self: Fitted pipeline
        """
        # Always use WorkflowManager and DocumentationEngine
        if self.workflow_manager is None:
            from src.workflows import WorkflowManager
            self.workflow_manager = WorkflowManager(user_query="classification workflow")
        if self.doc_engine is None:
            from src.workflows import DocumentationEngine
            self.doc_engine = DocumentationEngine(self.workflow_manager)

        # Define modular tasks for orchestrator
        def task_fit_model(X, y):
            step_id = "classification_pipeline_fit"
            self.workflow_manager.add_execution_step(
                step_id=step_id,
                macro_process_name="Classification",
                process_description="Fit classification model",
                responsible_agent_or_tool="AgentClassificationPipeline",
                inputs_used=[str(X.shape), str(y.shape)]
            )
            self.workflow_manager.start_execution_step(step_id)
            self.model.fit(X, y)
            self.is_fitted = True
            self.workflow_manager.complete_execution_step(
                step_id,
                message="Completed classification model fit",
                outputs=[str(X.shape), str(y.shape)]
            )
            self.doc_engine.generate_html_report()
            return "fit_complete"

        # Assemble orchestrator workflow
        from src.workflow_orchestrator.core import Workflow, Task, WorkflowEngine
        from src.workflow_orchestrator.utils import WorkflowBuilder

        workflow = WorkflowBuilder(
            name="ClassificationPipelineWorkflow",
            description="Agent-driven modular classification pipeline workflow"
        ).add_task(
            name="FitClassificationModel",
            function=lambda X=X, y=y: task_fit_model(X, y),
            parameters={"X": X, "y": y}
        ).build()

        # Register and run workflow
        engine = WorkflowEngine()
        engine.register_workflow(workflow)
        result = engine.run(workflow)

        # Document workflow execution (summary step)
        self.workflow_manager.add_execution_step(
            step_id="classification_workflow_completed",
            macro_process_name="Classification",
            process_description="Classification workflow completed",
            responsible_agent_or_tool="AgentClassificationPipeline",
            inputs_used=[str(X.shape), str(y.shape)]
        )
        self.workflow_manager.start_execution_step("classification_workflow_completed")
        self.workflow_manager.complete_execution_step(
            "classification_workflow_completed",
            message="Classification workflow completed",
            outputs=[str(X.shape), str(y.shape)]
        )
        self.doc_engine.generate_html_report()

        # DEPRECATED: Legacy direct logic (preserved for reference)
        # self.model.fit(X, y)
        # self.is_fitted = True

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions using the fitted model."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before making predictions")
        return self.model.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Get prediction probabilities (if supported)."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before making predictions")
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from the fitted model (if supported)."""
        if not self.is_fitted:
            return None
        if hasattr(self.model, 'get_feature_importance'):
            return self.model.get_feature_importance()
        return None

    def generate_report(self) -> str:
        """Generate a comprehensive classification report (if supported)."""
        if hasattr(self.model, 'generate_report'):
            return self.model.generate_report()
        return f"Model type: {self.model_type}\nFitted: {self.is_fitted}"
