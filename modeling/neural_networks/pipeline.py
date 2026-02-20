"""
AgentNeuralNetworkPipeline: Highly Adaptable, Agent-Driven, Workflow-Orchestrated, and Documented Neural Network Pipeline

This pipeline enables modular or full execution of neural network tasks by an agent, using workflow_orchestrator and workflows modules. The agent can assemble the workflow from config/capabilities and use the pipelines module in combination with workflow modules. Designed for maximum adaptability and modularity, following the pattern of src/preprocessing/.
"""

from typing import Optional, Dict, Any, List, Tuple, Union
from . import *

class AgentNeuralNetworkPipeline:
    """
    Highly adaptable neural network pipeline for agent-driven, workflow-orchestrated, and documented execution.
    Enables modular or full neural network tasks by an agent, using workflow_orchestrator and workflows modules.
    The agent can assemble the workflow from config/capabilities and use the pipelines module in combination with workflow modules.
    Example agent usage:
        - Agent looks in config/capabilities to assemble workflow steps and model parameters
        - Agent instantiates pipeline and executes run_workflow
        - All steps are documented using the workflows module
    """
    def __init__(self,
                 nn_model,
                 X,
                 y,
                 workflow_manager=None,
                 doc_engine=None,
                 agent_config: Optional[Dict[str, Any]] = None):
        self.nn_model = nn_model
        self.X = X
        self.y = y
        self.workflow_manager = workflow_manager
        self.doc_engine = doc_engine
        self.agent_config = agent_config or {}
        self.is_fitted = False


    def run_workflow(self, epochs: int = 100, batch_size: int = 32, validation_split: float = 0.2, workflow_manager=None, doc_engine=None):
        """
        Orchestrator-centric, agent-driven, and documented execution of neural network pipeline.
        Uses orchestrator Workflow, Task, WorkflowEngine, and always integrates WorkflowManager/DocumentationEngine.
        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of data for validation
        Returns:
            self: Fitted pipeline
        """
        from src.workflow_orchestrator.core import Workflow, Task, WorkflowEngine
        from src.workflow_orchestrator.utils import WorkflowBuilder
        if workflow_manager is None and self.workflow_manager is None:
            from src.workflows.workflow_manager import WorkflowManager
            workflow_manager = WorkflowManager(user_query="neural network workflow")
        if workflow_manager is None:
            workflow_manager = self.workflow_manager
        if doc_engine is None and self.doc_engine is None:
            from src.workflows.documentation_engine import DocumentationEngine
            doc_engine = DocumentationEngine(workflow_manager)
        if doc_engine is None:
            doc_engine = self.doc_engine

        # Define orchestrator tasks
        def task_fit(X, y, nn_model, epochs, batch_size, validation_split, workflow_context=None):
            step_id = "nn_fit"
            workflow_manager.add_execution_step(
                step_id=step_id,
                macro_process_name="NeuralNetworks",
                process_description="Fit neural network model",
                responsible_agent_or_tool="AgentNeuralNetworkPipeline",
                inputs_used=[str(X.shape), str(y.shape)]
            )
            workflow_manager.start_execution_step(step_id)
            # Only pass epochs, batch_size, validation_split if supported (e.g., Keras)
            import inspect
            fit_sig = inspect.signature(nn_model.fit)
            fit_params = fit_sig.parameters
            fit_kwargs = {}
            if 'epochs' in fit_params:
                fit_kwargs['epochs'] = epochs
            if 'batch_size' in fit_params:
                fit_kwargs['batch_size'] = batch_size
            if 'validation_split' in fit_params:
                fit_kwargs['validation_split'] = validation_split
            nn_model.fit(X, y, **fit_kwargs)
            self.is_fitted = True
            workflow_manager.complete_execution_step(step_id, message="Completed neural network fit", outputs=[str(X.shape), str(y.shape)])
            doc_engine.generate_html_report()
            return nn_model

        builder = WorkflowBuilder(
            name="NeuralNetworkPipelineWorkflow",
            description="Agent-driven modular neural network pipeline workflow"
        )
        t1 = builder.add_task(
            name="NeuralNetworkFit",
            function=lambda X=self.X, y=self.y, nn_model=self.nn_model, epochs=epochs, batch_size=batch_size, validation_split=validation_split, workflow_context=None: task_fit(X, y, nn_model, epochs, batch_size, validation_split, workflow_context),
            parameters={"X": self.X, "y": self.y, "nn_model": self.nn_model, "epochs": epochs, "batch_size": batch_size, "validation_split": validation_split}
        ).current_tasks[0].task_id
        workflow = builder.build()

        engine = WorkflowEngine()
        engine.register_workflow(workflow)
        result = engine.run(workflow)

        # Collect results from orchestrator context
        context = workflow.context
        # Optionally store model or results if needed

        # Document workflow execution (summary step)
        workflow_manager.add_execution_step(
            step_id="nn_pipeline_completed",
            macro_process_name="NeuralNetworks",
            process_description="Neural network pipeline completed",
            responsible_agent_or_tool="AgentNeuralNetworkPipeline",
            inputs_used=[str(self.X.shape), str(self.y.shape)]
        )
        workflow_manager.start_execution_step("nn_pipeline_completed")
        workflow_manager.complete_execution_step(
            "nn_pipeline_completed",
            message="Neural network pipeline completed",
            outputs=[str(self.X.shape), str(self.y.shape)]
        )
        doc_engine.generate_html_report()
        return self

    def predict(self, X):
        """Make predictions using the fitted neural network model."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before making predictions")
        return self.nn_model.predict(X)

    def get_model_summary(self) -> str:
        """Get model architecture summary (if supported)."""
        if hasattr(self.nn_model, 'get_model_summary'):
            return self.nn_model.get_model_summary()
        return f"Neural network pipeline fitted: {self.is_fitted}"

    def generate_report(self) -> str:
        """Generate a comprehensive neural network report (if supported)."""
        summary = self.get_model_summary()
        return f"Neural Network Pipeline Report\n{'='*40}\n{summary}"
