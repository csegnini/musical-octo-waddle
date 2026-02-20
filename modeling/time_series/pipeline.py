"""
AgentTimeSeriesPipeline: Highly Adaptable, Agent-Driven, Workflow-Orchestrated, and Documented Time Series Pipeline

This pipeline enables modular or full execution of time series tasks by an agent, using workflow_orchestrator and workflows modules. The agent can assemble the workflow from config/capabilities and use the pipelines module in combination with workflow modules. Designed for maximum adaptability and modularity, following the pattern of src/data_analysis.
"""

from typing import Optional, Dict, Any, List, Tuple, Union
from . import *

class AgentTimeSeriesPipeline:
    """
    Highly adaptable time series pipeline for agent-driven, workflow-orchestrated, and documented execution.
    Enables modular or full time series tasks by an agent, using workflow_orchestrator and workflows modules.
    The agent can assemble the workflow from config/capabilities and use the pipelines module in combination with workflow modules.
    Example agent usage:
        - Agent looks in config/capabilities to assemble workflow steps and model parameters
        - Agent instantiates pipeline and executes run_workflow
        - All steps are documented using the workflows module
    """
    def __init__(self,
                 ts_model,
                 X,
                 y=None,
                 workflow_manager=None,
                 doc_engine=None,
                 agent_config: Optional[Dict[str, Any]] = None):
        self.ts_model = ts_model
        self.X = X
        self.y = y
        self.workflow_manager = workflow_manager
        self.doc_engine = doc_engine
        self.agent_config = agent_config or {}
        self.is_fitted = False
        self.forecast_result = None

    def run_workflow(self, forecast_steps: int = 10, **fit_kwargs):
        """
        Agent-driven, orchestrator-based, and documented execution of time series pipeline.
        Uses orchestrator Workflow, Task, WorkflowEngine, and always integrates WorkflowManager/DocumentationEngine.
        Legacy direct logic is deprecated but preserved for reference.
        Args:
            forecast_steps: Number of steps to forecast
            fit_kwargs: Additional fit arguments for the time series model
        Returns:
            self: Fitted pipeline
        """
        # Always use WorkflowManager and DocumentationEngine
        if self.workflow_manager is None:
            from src.workflows.workflow_manager import WorkflowManager
            self.workflow_manager = WorkflowManager(user_query="time series workflow")
        if self.doc_engine is None:
            from src.workflows.documentation_engine import DocumentationEngine
            self.doc_engine = DocumentationEngine(self.workflow_manager)

        # Define modular task for orchestrator
        def task_fit_and_forecast(ts_model, X, forecast_steps, fit_kwargs):
            step_id = "ts_pipeline_fit_forecast"
            self.workflow_manager.add_execution_step(
                step_id=step_id,
                macro_process_name="TimeSeries",
                process_description="Fit and forecast time series model",
                responsible_agent_or_tool="AgentTimeSeriesPipeline",
                inputs_used=[str(X.shape), f"forecast_steps={forecast_steps}"]
            )
            self.workflow_manager.start_execution_step(step_id)
            if hasattr(ts_model, 'fit'):
                fitted_model = ts_model.fit(**fit_kwargs)
                if hasattr(fitted_model, 'forecast'):
                    self.forecast_result = fitted_model.forecast(steps=forecast_steps)
                elif hasattr(fitted_model, 'predict'):
                    self.forecast_result = fitted_model.predict(start=len(X), end=len(X)+forecast_steps-1)
                self.is_fitted = True
            else:
                raise ValueError("Provided time series model does not support fit method.")
            self.workflow_manager.complete_execution_step(
                step_id,
                message="Completed time series fit/forecast",
                outputs=[str(self.forecast_result)]
            )
            self.doc_engine.generate_html_report()
            return "fit_forecast_complete"

        # Assemble orchestrator workflow
        from src.workflow_orchestrator.core import Workflow, Task, WorkflowEngine
        from src.workflow_orchestrator.utils import WorkflowBuilder

        workflow = WorkflowBuilder(
            name="TimeSeriesPipelineWorkflow",
            description="Agent-driven modular time series pipeline workflow"
        ).add_task(
            name="FitAndForecastTimeSeriesModel",
            function=lambda ts_model=self.ts_model, X=self.X, forecast_steps=forecast_steps, fit_kwargs=fit_kwargs, **kwargs: task_fit_and_forecast(ts_model, X, forecast_steps, fit_kwargs),
            parameters={"ts_model": self.ts_model, "X": self.X, "forecast_steps": forecast_steps, "fit_kwargs": fit_kwargs}
        ).build()

        # Register and run workflow
        engine = WorkflowEngine()
        engine.register_workflow(workflow)
        result = engine.run(workflow)

        # Document workflow execution (summary step)
        self.workflow_manager.add_execution_step(
            step_id="ts_workflow_completed",
            macro_process_name="TimeSeries",
            process_description="Time series workflow completed",
            responsible_agent_or_tool="AgentTimeSeriesPipeline",
            inputs_used=[str(self.X.shape)]
        )
        self.workflow_manager.start_execution_step("ts_workflow_completed")
        self.workflow_manager.complete_execution_step(
            "ts_workflow_completed",
            message="Time series workflow completed",
            outputs=[str(self.forecast_result)]
        )
        self.doc_engine.generate_html_report()

        # DEPRECATED: Legacy direct logic (preserved for reference)
        # if hasattr(self.ts_model, 'fit'):
        #     fitted_model = self.ts_model.fit(**fit_kwargs)
        #     if hasattr(fitted_model, 'forecast'):
        #         self.forecast_result = fitted_model.forecast(steps=forecast_steps)
        #     elif hasattr(fitted_model, 'predict'):
        #         self.forecast_result = fitted_model.predict(start=len(self.X), end=len(self.X)+forecast_steps-1)
        #     self.is_fitted = True
        # else:
        #     raise ValueError("Provided time series model does not support fit method.")

        return self

    def get_report(self) -> str:
        """Return a formatted time series pipeline report."""
        return f"Time Series Pipeline Report\n{'='*40}\nForecast: {self.forecast_result}"
