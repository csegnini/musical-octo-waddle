"""
Deep Learning Pipeline for Agent-Driven, Workflow-Orchestrated, and Documented Execution

This module provides a highly adaptable deep learning pipeline class that can be orchestrated by an agent using the workflow_orchestrator and documented using the workflows module. The pipeline can be assembled from config/capabilities and supports integration with the agent, WorkflowManager, DocumentationEngine, and other workflow modules.
"""

from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
import pandas as pd
from . import (
    ConvolutionalNetwork, RecurrentNetwork, TransformerNetwork, VisionTransformer,
    create_cnn_classifier, create_rnn_classifier, create_transformer_model, create_vision_transformer,
    CNNArchitecture, RNNType
)

class AgentDeepLearningPipeline:
    """
    Highly adaptable deep learning pipeline for agent-driven, workflow-orchestrated, and documented execution.
    """
    def __init__(self,
                 model_type: str = 'cnn',
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
        if self.model_type == 'cnn':
            return create_cnn_classifier(**self.model_params)
        elif self.model_type == 'rnn':
            return create_rnn_classifier(**self.model_params)
        elif self.model_type == 'transformer':
            return create_transformer_model(**self.model_params)
        elif self.model_type == 'vit':
            return create_vision_transformer(**self.model_params)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def run_workflow(self, X, y, validation_data=None, epochs=10, batch_size=32) -> 'AgentDeepLearningPipeline':
        """
        Agent-driven, orchestrator-based, and documented execution of deep learning pipeline.
        Uses orchestrator Workflow, Task, WorkflowEngine, and always integrates WorkflowManager/DocumentationEngine.
        Legacy direct logic is deprecated but preserved for reference.
        Args:
            X: Input features
            y: Target variable
            validation_data: Optional validation data
            epochs: Number of training epochs
            batch_size: Training batch size
        Returns:
            self: Fitted pipeline
        """
        # Always use WorkflowManager and DocumentationEngine
        if self.workflow_manager is None:
            from src.workflows import WorkflowManager
            self.workflow_manager = WorkflowManager(user_query="deep learning workflow")
        if self.doc_engine is None:
            from src.workflows import DocumentationEngine
            self.doc_engine = DocumentationEngine(self.workflow_manager)

        # Define modular tasks for orchestrator
        def task_fit_deep_learning_model(X, y, validation_data, epochs, batch_size):
            step_id = "deep_learning_pipeline_fit"
            self.workflow_manager.add_execution_step(
                step_id=step_id,
                macro_process_name="DeepLearning",
                process_description="Fit deep learning model",
                responsible_agent_or_tool="AgentDeepLearningPipeline",
                inputs_used=[str(np.shape(X)), str(np.shape(y)), f"epochs={epochs}", f"batch_size={batch_size}"]
            )
            self.workflow_manager.start_execution_step(step_id)
            self.model.fit(X, y, validation_data=validation_data, epochs=epochs, batch_size=batch_size)
            self.is_fitted = True
            self.workflow_manager.complete_execution_step(
                step_id,
                message="Completed deep learning model fit",
                outputs=[str(np.shape(X)), str(np.shape(y))]
            )
            self.doc_engine.generate_html_report()
            return "fit_complete"

        # Assemble orchestrator workflow
        from src.workflow_orchestrator.core import Workflow, Task, WorkflowEngine
        from src.workflow_orchestrator.utils import WorkflowBuilder

        workflow = WorkflowBuilder(
            name="DeepLearningPipelineWorkflow",
            description="Agent-driven modular deep learning pipeline workflow"
        ).add_task(
            name="FitDeepLearningModel",
            function=lambda X=X, y=y, validation_data=validation_data, epochs=epochs, batch_size=batch_size: task_fit_deep_learning_model(X, y, validation_data, epochs, batch_size),
            parameters={"X": X, "y": y, "validation_data": validation_data, "epochs": epochs, "batch_size": batch_size}
        ).build()

        # Register and run workflow
        engine = WorkflowEngine()
        engine.register_workflow(workflow)
        result = engine.run(workflow)

        # Document workflow execution (summary step)
        self.workflow_manager.add_execution_step(
            step_id="deep_learning_workflow_completed",
            macro_process_name="DeepLearning",
            process_description="Deep learning workflow completed",
            responsible_agent_or_tool="AgentDeepLearningPipeline",
            inputs_used=[str(np.shape(X)), str(np.shape(y))]
        )
        self.workflow_manager.start_execution_step("deep_learning_workflow_completed")
        self.workflow_manager.complete_execution_step(
            "deep_learning_workflow_completed",
            message="Deep learning workflow completed",
            outputs=[str(np.shape(X)), str(np.shape(y))]
        )
        self.doc_engine.generate_html_report()

        # DEPRECATED: Legacy direct logic (preserved for reference)
        # self.model.fit(X, y, validation_data=validation_data, epochs=epochs, batch_size=batch_size)
        # self.is_fitted = True

        return self

    def predict(self, X):
        """Make predictions using the fitted model."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before making predictions")
        return self.model.predict(X)

    def predict_proba(self, X):
        """Get prediction probabilities (if supported)."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before making predictions")
        return self.model.predict_proba(X)

    def get_model_summary(self) -> str:
        """Get model architecture summary (if supported)."""
        if hasattr(self.model, 'get_model_summary'):
            return self.model.get_model_summary()
        return f"Model type: {self.model_type}\nFitted: {self.is_fitted}"

    def generate_report(self) -> str:
        """Generate a comprehensive deep learning report (if supported)."""
        summary = self.get_model_summary()
        return f"Deep Learning Pipeline Report\n{'='*40}\n{summary}"
