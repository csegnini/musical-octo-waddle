"""
AgentRegressionPipeline: Highly Adaptable, Agent-Driven, Workflow-Orchestrated, and Documented Regression Pipeline

This pipeline enables modular or full execution of regression tasks by an agent, using workflow_orchestrator and workflows modules. The agent can assemble the workflow from config/capabilities and use the pipelines module in combination with workflow modules. Designed for maximum adaptability and modularity, following the pattern of src/preprocessing/.
"""

from typing import Optional, Dict, Any, List, Tuple, Union
from . import *

class AgentRegressionPipeline:
    """
    Highly adaptable regression pipeline for agent-driven, workflow-orchestrated, and documented execution.
    Enables modular or full regression tasks by an agent, using workflow_orchestrator and workflows modules.
    The agent can assemble the workflow from config/capabilities and use the pipelines module in combination with workflow modules.
    Example agent usage:
        - Agent looks in config/capabilities to assemble workflow steps and model parameters
        - Agent instantiates pipeline and executes run_workflow
        - All steps are documented using the workflows module
    """
    def __init__(self,
                 reg_model,
                 X,
                 y,
                 workflow_manager=None,
                 doc_engine=None,
                 agent_config: Optional[Dict[str, Any]] = None):
        self.reg_model = reg_model
        self.X = X
        self.y = y
        self.workflow_manager = workflow_manager
        self.doc_engine = doc_engine
        self.agent_config = agent_config or {}
        self.is_fitted = False

    def run_workflow(self, test_size: float = 0.2):
        """
        Agent-driven, orchestrator-based, and documented execution of regression pipeline.
        Uses orchestrator Workflow, Task, WorkflowEngine, and always integrates WorkflowManager/DocumentationEngine.
        Legacy direct logic is deprecated but preserved for reference.
        Args:
            test_size: Fraction of data for testing
        Returns:
            self: Fitted pipeline
        """
        # Always use WorkflowManager and DocumentationEngine
        if self.workflow_manager is None:
            from src.workflows.workflow_manager import WorkflowManager
            self.workflow_manager = WorkflowManager(user_query="regression workflow")
        if self.doc_engine is None:
            from src.workflows.documentation_engine import DocumentationEngine
            self.doc_engine = DocumentationEngine(self.workflow_manager)

        # Define modular tasks for orchestrator
        def task_fit_regression_model(X, y, test_size):
            step_id = "regression_pipeline_fit"
            self.workflow_manager.add_execution_step(
                step_id=step_id,
                macro_process_name="Regression",
                process_description="Fit regression model",
                responsible_agent_or_tool="AgentRegressionPipeline",
                inputs_used=[str(X.shape), str(y.shape), f"test_size={test_size}"]
            )
            self.workflow_manager.start_execution_step(step_id)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            self.reg_model.fit(X_train, y_train)
            self.is_fitted = True
            self.X_test = X_test
            self.y_test = y_test
            self.workflow_manager.complete_execution_step(
                step_id,
                message="Completed regression model fit",
                outputs=[str(X.shape), str(y.shape)]
            )
            self.doc_engine.generate_html_report()
            return "fit_complete"

        # Assemble orchestrator workflow
        from src.workflow_orchestrator.core import Workflow, Task, WorkflowEngine
        from src.workflow_orchestrator.utils import WorkflowBuilder

        workflow = WorkflowBuilder(
            name="RegressionPipelineWorkflow",
            description="Agent-driven modular regression pipeline workflow"
        ).add_task(
            name="FitRegressionModel",
            function=lambda **kwargs: task_fit_regression_model(self.X, self.y, test_size),
            parameters={"X": self.X, "y": self.y, "test_size": test_size}
        ).build()

        # Register and run workflow
        engine = WorkflowEngine()
        engine.register_workflow(workflow)
        result = engine.run(workflow)

        # Document workflow execution (summary step)
        self.workflow_manager.add_execution_step(
            step_id="regression_workflow_completed",
            macro_process_name="Regression",
            process_description="Regression workflow completed",
            responsible_agent_or_tool="AgentRegressionPipeline",
            inputs_used=[str(self.X.shape), str(self.y.shape)]
        )
        self.workflow_manager.start_execution_step("regression_workflow_completed")
        self.workflow_manager.complete_execution_step(
            "regression_workflow_completed",
            message="Regression workflow completed",
            outputs=[str(self.X.shape), str(self.y.shape)]
        )
        self.doc_engine.generate_html_report()

        # DEPRECATED: Legacy direct logic (preserved for reference)
        # from sklearn.model_selection import train_test_split
        # X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        # self.reg_model.fit(X_train, y_train)
        # self.is_fitted = True
        # self.X_test = X_test
        # self.y_test = y_test

        return self

    def predict(self, X):
        """Make predictions using the fitted regression model."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before making predictions")
        return self.reg_model.predict(X)

    def get_model_summary(self) -> str:
        """Get model architecture summary (if supported)."""
        if hasattr(self.reg_model, 'get_model_summary'):
            return self.reg_model.get_model_summary()
        return f"Regression pipeline fitted: {self.is_fitted}"

    def generate_report(self) -> str:
        """Generate a comprehensive regression report (if supported)."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        if not self.is_fitted:
            return "Model not fitted."
        y_pred = self.reg_model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        summary = self.get_model_summary()
        report = f"Regression Pipeline Report\n{'='*40}\n{summary}\nMSE: {mse:.4f}\nMAE: {mae:.4f}\nR2: {r2:.4f}"
        return report
