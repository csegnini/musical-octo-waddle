from typing import Optional, Dict, Any, List, Tuple, Union
from . import *

class AgentEvaluationPipeline:
    """
    Highly adaptable evaluation pipeline for agent-driven, workflow-orchestrated, and documented execution.
    Enables modular or full evaluation tasks by an agent, using workflow_orchestrator and workflows modules.
    The agent can assemble the workflow from config/capabilities and use the pipelines module in combination with workflow modules.
    """
    def __init__(self,
                 model,
                 X,
                 y,
                 model_type: 'ModelType',
                 workflow_manager=None,
                 doc_engine=None,
                 agent_config: Optional[Dict[str, Any]] = None):
        self.model = model
        self.X = X
        self.y = y
        self.model_type = model_type
        self.workflow_manager = workflow_manager
        self.doc_engine = doc_engine
        self.agent_config = agent_config or {}
        self.evaluation_result = None

    def run_workflow(self, test_size: float = 0.2):
        """
        Agent-driven, orchestrator-based, and documented execution of evaluation pipeline.
        Uses orchestrator Workflow, Task, WorkflowEngine, and always integrates WorkflowManager/DocumentationEngine.
        Legacy direct logic is deprecated but preserved for reference.
        Args:
            test_size: Test set size for evaluation
        Returns:
            self: Fitted pipeline
        """
        # Always use WorkflowManager and DocumentationEngine
        if self.workflow_manager is None:
            from src.workflows.workflow_manager import WorkflowManager
            self.workflow_manager = WorkflowManager(user_query="evaluation workflow")
        if self.doc_engine is None:
            from src.workflows.documentation_engine import DocumentationEngine
            self.doc_engine = DocumentationEngine(self.workflow_manager)

        # Define modular tasks for orchestrator
        def task_evaluate_model(model, X, y, model_type, test_size):
            step_id = "evaluation_pipeline_evaluate"
            self.workflow_manager.add_execution_step(
                step_id=step_id,
                macro_process_name="Evaluation",
                process_description="Evaluate model",
                responsible_agent_or_tool="AgentEvaluationPipeline",
                inputs_used=[str(model), str(X.shape), str(y.shape), f"test_size={test_size}"]
            )
            self.workflow_manager.start_execution_step(step_id)
            suite = EvaluationSuite()
            self.evaluation_result = suite.evaluate_model(model, X, y, model_type, test_size=test_size)
            self.workflow_manager.complete_execution_step(
                step_id,
                message="Completed model evaluation",
                outputs=[str(self.evaluation_result.metrics)]
            )
            self.doc_engine.generate_html_report()
            return "evaluation_complete"

        # Assemble orchestrator workflow
        from src.workflow_orchestrator.core import Workflow, Task, WorkflowEngine
        from src.workflow_orchestrator.utils import WorkflowBuilder

        workflow = WorkflowBuilder(
            name="EvaluationPipelineWorkflow",
            description="Agent-driven modular evaluation pipeline workflow"
        ).add_task(
            name="EvaluateModel",
            function=lambda model=self.model, X=self.X, y=self.y, model_type=self.model_type, test_size=test_size: task_evaluate_model(model, X, y, model_type, test_size),
            parameters={"model": self.model, "X": self.X, "y": self.y, "model_type": self.model_type, "test_size": test_size}
        ).build()

        # Register and run workflow
        engine = WorkflowEngine()
        engine.register_workflow(workflow)
        result = engine.run(workflow)

        # Document workflow execution (summary step)
        self.workflow_manager.add_execution_step(
            step_id="evaluation_workflow_completed",
            macro_process_name="Evaluation",
            process_description="Evaluation workflow completed",
            responsible_agent_or_tool="AgentEvaluationPipeline",
            inputs_used=[str(self.model), str(self.X.shape), str(self.y.shape)]
        )
        self.workflow_manager.start_execution_step("evaluation_workflow_completed")
        self.workflow_manager.complete_execution_step(
            "evaluation_workflow_completed",
            message="Evaluation workflow completed",
            outputs=[str(self.evaluation_result.metrics)]
        )
        self.doc_engine.generate_html_report()

        # DEPRECATED: Legacy direct logic (preserved for reference)
        # suite = EvaluationSuite()
        # self.evaluation_result = suite.evaluate_model(self.model, self.X, self.y, self.model_type, test_size=test_size)

        return self

    def get_report(self) -> str:
        """Return a formatted evaluation report."""
        if self.evaluation_result is None:
            return "No evaluation performed yet."
        if self.model_type == ModelType.REGRESSION:
            # Format metrics dict as string
            return '\n'.join(f"{k}: {v}" for k, v in self.evaluation_result.metrics.items())
        else:
            return self.evaluation_result.classification_report or "No classification report available."

    def get_metrics(self) -> Dict[str, float]:
        """Return evaluation metrics."""
        if self.evaluation_result is None:
            return {}
        return self.evaluation_result.metrics