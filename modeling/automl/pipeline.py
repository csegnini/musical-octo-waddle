"""
AutoML Pipeline for Agent-Driven, Workflow-Orchestrated, and Documented Execution

This module provides a highly adaptable AutoML pipeline class that can be orchestrated by an agent using the workflow_orchestrator and documented using the workflows module. The pipeline can be assembled from config/capabilities and supports integration with the agent, WorkflowManager, DocumentationEngine, and other workflow modules.
"""

from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from . import AutoMLPipeline, AutoMLConfig, AutoMLMode, OptimizationAlgorithm, FeatureSelectionMethod

class AgentAutoMLPipeline:
    """
    Highly adaptable AutoML pipeline for agent-driven, workflow-orchestrated, and documented execution.
    """
    def __init__(self,
                 config: Optional[AutoMLConfig] = None,
                 workflow_manager=None,
                 doc_engine=None,
                 agent_config: Optional[Dict[str, Any]] = None):
        self.config = config or AutoMLConfig()
        self.workflow_manager = workflow_manager
        self.doc_engine = doc_engine
        self.agent_config = agent_config or {}
        self.pipeline = AutoMLPipeline(self.config)
        self.results = None

    def run_workflow(self, X: np.ndarray, y: np.ndarray) -> 'AgentAutoMLPipeline':
        """
        Agent-driven, workflow-orchestrated, and documented execution of AutoML pipeline.
        Each step is documented using the workflows module. Highly adaptable for agent/capabilities/config integration.
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target variable
        Returns:
            self: Fitted pipeline
        """
        if self.workflow_manager is None:
            from src.workflows import WorkflowManager
            self.workflow_manager = WorkflowManager(user_query="automl workflow")
        if self.doc_engine is None:
            from src.workflows import DocumentationEngine
            self.doc_engine = DocumentationEngine(self.workflow_manager)

        # Document pipeline start
        self.workflow_manager.add_execution_step(
            step_id="automl_pipeline_start",
            macro_process_name="AutoML",
            process_description="Start AutoML pipeline",
            responsible_agent_or_tool="AgentAutoMLPipeline",
            inputs_used=[str(X.shape), str(y.shape)]
        )
        self.workflow_manager.start_execution_step("automl_pipeline_start")
        self.doc_engine.generate_html_report()

        # Fit AutoML pipeline
        self.results = self.pipeline.fit(X, y)

        # Document pipeline end
        self.workflow_manager.complete_execution_step("automl_pipeline_start", message="Completed AutoML pipeline", outputs=[str(X.shape), str(y.shape)])
        self.doc_engine.generate_html_report()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the best model."""
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities (classification only)."""
        return self.pipeline.predict_proba(X)

    def get_model_rankings(self) -> List[Tuple[str, float]]:
        """Get models ranked by performance."""
        return self.pipeline.get_model_rankings()

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from the best model."""
        return self.pipeline.get_feature_importance()

    def generate_report(self) -> str:
        """Generate a comprehensive AutoML report."""
        return self.pipeline.generate_report()
