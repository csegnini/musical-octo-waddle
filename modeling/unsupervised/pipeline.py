"""
AgentUnsupervisedPipeline: Highly Adaptable, Agent-Driven, Workflow-Orchestrated, and Documented Unsupervised Learning Pipeline

This pipeline enables modular or full execution of unsupervised learning tasks by an agent, using workflow_orchestrator and workflows modules. The agent can assemble the workflow from config/capabilities and use the pipelines module in combination with workflow modules. Designed for maximum adaptability and modularity, following the pattern of src/data_analysis.
"""

from typing import Optional, Dict, Any, List, Tuple, Union
from . import *

class AgentUnsupervisedPipeline:
    """
    Highly adaptable unsupervised learning pipeline for agent-driven, workflow-orchestrated, and documented execution.
    Enables modular or full unsupervised learning tasks by an agent, using workflow_orchestrator and workflows modules.
    The agent can assemble the workflow from config/capabilities and use the pipelines module in combination with workflow modules.
    Example agent usage:
        - Agent looks in config/capabilities to assemble workflow steps and model parameters
        - Agent instantiates pipeline and executes run_workflow
        - All steps are documented using the workflows module
    """
    def __init__(self,
                 unsup_model,
                 X,
                 workflow_manager=None,
                 doc_engine=None,
                 agent_config: Optional[Dict[str, Any]] = None):
        self.unsup_model = unsup_model
        self.X = X
        self.workflow_manager = workflow_manager
        self.doc_engine = doc_engine
        self.agent_config = agent_config or {}
        self.is_fitted = False
        self.result = None


    def run_workflow(self, fit_kwargs: Optional[Dict[str, any]] = None, workflow_manager=None, doc_engine=None):
        """
        Orchestrator-centric, agent-driven, and documented execution of unsupervised pipeline.
        Uses orchestrator Workflow, Task, WorkflowEngine, and always integrates WorkflowManager/DocumentationEngine.
        Args:
            fit_kwargs: Additional fit arguments for the unsupervised model
        Returns:
            self: Fitted pipeline
        """
        from src.workflow_orchestrator.core import Workflow, Task, WorkflowEngine
        from src.workflow_orchestrator.utils import WorkflowBuilder
        if workflow_manager is None and self.workflow_manager is None:
            from src.workflows.workflow_manager import WorkflowManager
            workflow_manager = WorkflowManager(user_query="unsupervised workflow")
        if workflow_manager is None:
            workflow_manager = self.workflow_manager
        if doc_engine is None and self.doc_engine is None:
            from src.workflows.documentation_engine import DocumentationEngine
            doc_engine = DocumentationEngine(workflow_manager)
        if doc_engine is None:
            doc_engine = self.doc_engine
        fit_kwargs = fit_kwargs or {}

        # Define orchestrator tasks
        def task_fit(X, unsup_model, fit_kwargs, workflow_context=None):
            step_id = "unsup_fit"
            workflow_manager.add_execution_step(
                step_id=step_id,
                macro_process_name="Unsupervised",
                process_description="Fit unsupervised model",
                responsible_agent_or_tool="AgentUnsupervisedPipeline",
                inputs_used=[str(X.shape)]
            )
            workflow_manager.start_execution_step(step_id)
            result = unsup_model.fit(X, **fit_kwargs)
            workflow_manager.complete_execution_step(step_id, message="Completed unsupervised fit", outputs=[str(result)])
            doc_engine.generate_html_report()
            return result

        builder = WorkflowBuilder(
            name="UnsupervisedPipelineWorkflow",
            description="Agent-driven modular unsupervised learning pipeline workflow"
        )
        t1 = builder.add_task(
            name="UnsupervisedFit",
            function=lambda X=self.X, unsup_model=self.unsup_model, fit_kwargs=fit_kwargs, workflow_context=None, **kwargs: task_fit(X, unsup_model, fit_kwargs, workflow_context),
            parameters={"X": self.X, "unsup_model": self.unsup_model, "fit_kwargs": fit_kwargs}
        ).current_tasks[0].task_id
        workflow = builder.build()

        engine = WorkflowEngine()
        engine.register_workflow(workflow)
        result = engine.run(workflow)

        # Collect results from orchestrator result object
        self.result = result.task_results[t1].output if t1 in result.task_results else None
        self.is_fitted = True

        # Document workflow execution (summary step)
        workflow_manager.add_execution_step(
            step_id="unsup_pipeline_completed",
            macro_process_name="Unsupervised",
            process_description="Unsupervised pipeline completed",
            responsible_agent_or_tool="AgentUnsupervisedPipeline",
            inputs_used=[str(self.X.shape)]
        )
        workflow_manager.start_execution_step("unsup_pipeline_completed")
        workflow_manager.complete_execution_step(
            "unsup_pipeline_completed",
            message="Unsupervised pipeline completed",
            outputs=[str(self.result)]
        )
        doc_engine.generate_html_report()
        return self

    def get_report(self) -> str:
        """Return a formatted unsupervised pipeline report."""
        return f"Unsupervised Pipeline Report\n{'='*40}\nResult: {self.result}"
