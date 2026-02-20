"""
AgentStatisticalAnalysisPipeline: Highly Adaptable, Agent-Driven, Workflow-Orchestrated, and Documented Statistical Analysis Pipeline

This pipeline enables modular or full execution of statistical analysis tasks by an agent, using workflow_orchestrator and workflows modules. The agent can assemble the workflow from config/capabilities and use the pipelines module in combination with workflow modules. Designed for maximum adaptability and modularity, following the pattern of src/preprocessing/.
"""

from typing import Optional, Dict, Any, List, Tuple, Union
from . import *

class AgentStatisticalAnalysisPipeline:
    """
    Highly adaptable statistical analysis pipeline for agent-driven, workflow-orchestrated, and documented execution.
    Enables modular or full statistical analysis tasks by an agent, using workflow_orchestrator and workflows modules.
    The agent can assemble the workflow from config/capabilities and use the pipelines module in combination with workflow modules.
    Example agent usage:
        - Agent looks in config/capabilities to assemble workflow steps and analysis parameters
        - Agent instantiates pipeline and executes run_workflow
        - All steps are documented using the workflows module
    """
    def __init__(self,
                 analysis_func,
                 X,
                 y=None,
                 workflow_manager=None,
                 doc_engine=None,
                 agent_config: Optional[Dict[str, Any]] = None):
        self.analysis_func = analysis_func
        self.X = X
        self.y = y
        self.workflow_manager = workflow_manager
        self.doc_engine = doc_engine
        self.agent_config = agent_config or {}
        self.analysis_result = None


    def run_workflow(self, analysis_kwargs: Optional[Dict[str, any]] = None, workflow_manager=None, doc_engine=None):
        """
        Orchestrator-centric, agent-driven, and documented execution of statistical analysis pipeline.
        Uses orchestrator Workflow, Task, WorkflowEngine, and always integrates WorkflowManager/DocumentationEngine.
        Args:
            analysis_kwargs: Keyword arguments for the analysis function
        Returns:
            self: Pipeline with analysis result
        """
        from src.workflow_orchestrator.core import Workflow, Task, WorkflowEngine
        from src.workflow_orchestrator.utils import WorkflowBuilder
        if workflow_manager is None and self.workflow_manager is None:
            from src.workflows.workflow_manager import WorkflowManager
            workflow_manager = WorkflowManager(user_query="statistical analysis workflow")
        if workflow_manager is None:
            workflow_manager = self.workflow_manager
        if doc_engine is None and self.doc_engine is None:
            from src.workflows.documentation_engine import DocumentationEngine
            doc_engine = DocumentationEngine(workflow_manager)
        if doc_engine is None:
            doc_engine = self.doc_engine
        analysis_kwargs = analysis_kwargs or {}

        # Define orchestrator tasks
        def task_analysis(X, y, analysis_func, analysis_kwargs, workflow_context=None):
            step_id = "stat_analysis_fit"
            workflow_manager.add_execution_step(
                step_id=step_id,
                macro_process_name="StatisticalAnalysis",
                process_description="Run statistical analysis",
                responsible_agent_or_tool="AgentStatisticalAnalysisPipeline",
                inputs_used=[str(X.shape)]
            )
            workflow_manager.start_execution_step(step_id)
            if y is not None:
                result = analysis_func(X, y, **analysis_kwargs)
            else:
                result = analysis_func(X, **analysis_kwargs)
            print("DEBUG: result =", result)
            workflow_manager.complete_execution_step(step_id, message="Completed statistical analysis", outputs=[str(result)])
            doc_engine.generate_html_report()
            return result

        builder = WorkflowBuilder(
            name="StatisticalAnalysisPipelineWorkflow",
            description="Agent-driven modular statistical analysis pipeline workflow"
        )
        t1 = builder.add_task(
            name="StatisticalAnalysisFit",
            function=lambda X=self.X, y=self.y, analysis_func=self.analysis_func, analysis_kwargs=analysis_kwargs, workflow_context=None, **kwargs: task_analysis(X, y, analysis_func, analysis_kwargs, workflow_context),
            parameters={"X": self.X, "y": self.y, "analysis_func": self.analysis_func, "analysis_kwargs": analysis_kwargs}
        ).current_tasks[0].task_id
        workflow = builder.build()

        engine = WorkflowEngine()
        engine.register_workflow(workflow)

        result = engine.run(workflow)
        print("DEBUG: workflow.context =", workflow.context)

        # Collect results from orchestrator result object
        self.analysis_result = result.task_results[t1].output if t1 in result.task_results else None

        # Document workflow execution (summary step)
        workflow_manager.add_execution_step(
            step_id="stat_analysis_pipeline_completed",
            macro_process_name="StatisticalAnalysis",
            process_description="Statistical analysis pipeline completed",
            responsible_agent_or_tool="AgentStatisticalAnalysisPipeline",
            inputs_used=[str(self.X.shape)]
        )
        workflow_manager.start_execution_step("stat_analysis_pipeline_completed")
        workflow_manager.complete_execution_step(
            "stat_analysis_pipeline_completed",
            message="Statistical analysis pipeline completed",
            outputs=[str(self.analysis_result)]
        )
        doc_engine.generate_html_report()
        return self

    def get_report(self) -> str:
        """Return a formatted statistical analysis report."""
        return f"Statistical Analysis Pipeline Report\n{'='*40}\nResult: {self.analysis_result}"
