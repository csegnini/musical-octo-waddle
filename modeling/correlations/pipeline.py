"""
AgentCorrelationsPipeline: Highly Adaptable, Agent-Driven, Workflow-Orchestrated, and Documented Correlation Analysis Pipeline

This pipeline enables modular or full execution of correlation analysis tasks by an agent, using workflow_orchestrator and workflows modules. The agent can assemble the workflow from config/capabilities and use the pipelines module in combination with workflow modules. Designed for maximum adaptability and modularity, following the orchestrator-centric pattern.
"""

import pandas as pd
from typing import Optional, Dict, Any
from src.correlations.config import CorrelationConfig
from src.correlations.algorithms import CorrelationAlgorithms
from src.correlations.rolling_analyzer import RollingCorrelationAnalyzer
from src.correlations.advanced_analyzer import AdvancedCorrelationAnalyzer
from src.correlations.visualizer import CorrelationVisualizer

class AgentCorrelationsPipeline:
    """
    Highly adaptable correlations pipeline for agent-driven, workflow-orchestrated, and documented execution.
    Enables modular or full correlation analysis tasks by an agent, using workflow_orchestrator and workflows modules.
    The agent can assemble the workflow from config/capabilities and use the pipelines module in combination with workflow modules.
    """
    def __init__(self, config: Optional[CorrelationConfig] = None, workflow_manager=None, doc_engine=None, agent_config: Optional[Dict[str, Any]] = None):
        self.config = config or CorrelationConfig()
        self.algorithms = CorrelationAlgorithms(self.config)
        self.rolling_analyzer = RollingCorrelationAnalyzer(self.config)
        self.advanced_analyzer = AdvancedCorrelationAnalyzer(self.config)
        self.visualizer = CorrelationVisualizer(self.config)
        self.workflow_manager = workflow_manager
        self.doc_engine = doc_engine
        self.agent_config = agent_config or {}
        self._results = {}

    def run_workflow(self, data: pd.DataFrame, **kwargs):
        """
        Agent-driven, orchestrator-based, and documented execution of correlations pipeline.
        Uses orchestrator Workflow, Task, WorkflowEngine, and always integrates WorkflowManager/DocumentationEngine.
        Legacy direct logic is deprecated but preserved for reference.
        Args:
            data: Input data
            kwargs: Additional arguments for pipeline steps
        Returns:
            self: Fitted pipeline
        """
        # Always use WorkflowManager and DocumentationEngine
        if self.workflow_manager is None:
            from src.workflows.workflow_manager import WorkflowManager
            self.workflow_manager = WorkflowManager(user_query="correlation analysis workflow")
        if self.doc_engine is None:
            from src.workflows.documentation_engine import DocumentationEngine
            self.doc_engine = DocumentationEngine(self.workflow_manager)

        from src.workflow_orchestrator.core import Workflow, Task, WorkflowEngine
        from src.workflow_orchestrator.utils import WorkflowBuilder

        def task_basic_correlations(data):
            step_id = "basic_correlation_analysis"
            self.workflow_manager.add_execution_step(
                step_id=step_id,
                macro_process_name="Correlation Analysis",
                process_description="Run basic correlation analysis",
                responsible_agent_or_tool="AgentCorrelationsPipeline",
                inputs_used=[str(data.shape)]
            )
            self.workflow_manager.start_execution_step(step_id)
            result = self.algorithms.calculate_all_correlations(data)
            self.workflow_manager.complete_execution_step(step_id, message="Completed basic correlation analysis", outputs=[str(result)])
            self.doc_engine.generate_html_report()
            return result

        def task_rolling_analysis(data):
            step_id = "rolling_correlation_analysis"
            self.workflow_manager.add_execution_step(
                step_id=step_id,
                macro_process_name="Correlation Analysis",
                process_description="Run rolling correlation analysis",
                responsible_agent_or_tool="AgentCorrelationsPipeline",
                inputs_used=[str(data.shape)]
            )
            self.workflow_manager.start_execution_step(step_id)
            result = self.rolling_analyzer.calculate_rolling_correlations(data)
            self.workflow_manager.complete_execution_step(step_id, message="Completed rolling correlation analysis", outputs=[str(result)])
            self.doc_engine.generate_html_report()
            return result

        def task_advanced_analysis(data):
            step_id = "advanced_correlation_analysis"
            self.workflow_manager.add_execution_step(
                step_id=step_id,
                macro_process_name="Correlation Analysis",
                process_description="Run advanced correlation analysis",
                responsible_agent_or_tool="AgentCorrelationsPipeline",
                inputs_used=[str(data.shape)]
            )
            self.workflow_manager.start_execution_step(step_id)
            # Run several advanced analyses and aggregate results
            granger = self.advanced_analyzer.granger_causality_analysis(data)
            cointegration = self.advanced_analyzer.cointegration_analysis(data)
            lead_lag = self.advanced_analyzer.lead_lag_analysis(data)
            regime = self.advanced_analyzer.regime_aware_correlation(data)
            result = {
                'granger_causality': granger,
                'cointegration': cointegration,
                'lead_lag': lead_lag,
                'regime_aware': regime
            }
            self.workflow_manager.complete_execution_step(step_id, message="Completed advanced correlation analysis", outputs=[str(result)])
            self.doc_engine.generate_html_report()
            return result

        def task_visualizations(basic_results, rolling_results, advanced_results):
            step_id = "correlation_visualization"
            self.workflow_manager.add_execution_step(
                step_id=step_id,
                macro_process_name="Correlation Analysis",
                process_description="Generate correlation visualizations",
                responsible_agent_or_tool="AgentCorrelationsPipeline",
                inputs_used=["basic, rolling, advanced results"]
            )
            self.workflow_manager.start_execution_step(step_id)
            # Visualize main correlation matrix (use first available from basic_results)
            main_corr = None
            if isinstance(basic_results, dict):
                for v in basic_results.values():
                    if isinstance(v, pd.DataFrame):
                        main_corr = v
                        break
            fig_matrix = self.visualizer.plot_correlation_matrix(main_corr) if main_corr is not None else None
            # Visualize rolling correlations (concatenate all DataFrames from rolling_results)
            rolling_df = None
            if isinstance(rolling_results, dict):
                dfs = [v for v in rolling_results.values() if isinstance(v, pd.DataFrame) and not v.empty]
                if dfs:
                    rolling_df = pd.concat(dfs, ignore_index=True)
                    # If the DataFrame is not in long format, convert it
                    if not set(['variable_1', 'variable_2', 'correlation', 'timestamp']).issubset(rolling_df.columns):
                        # Assume index is MultiIndex (timestamp, variable_1), columns are variable_2
                        if isinstance(rolling_df.index, pd.MultiIndex) and len(rolling_df.index.names) == 2:
                            rolling_df = rolling_df.stack().reset_index()
                            rolling_df.columns = ['timestamp', 'variable_1', 'variable_2', 'correlation']
                        else:
                            # Fallback: try to melt if index is timestamp and columns are variable_2
                            rolling_df = rolling_df.reset_index().melt(id_vars=['index'], var_name='variable_2', value_name='correlation')
                            rolling_df.rename(columns={'index': 'timestamp'}, inplace=True)
                            rolling_df['variable_1'] = None  # Not enough info, but prevents crash
            fig_rolling = self.visualizer.plot_rolling_correlations(rolling_df) if rolling_df is not None and not rolling_df.empty else None
            # Visualize comparison of all basic correlation matrices
            fig_comparison = self.visualizer.plot_correlation_comparison(basic_results) if isinstance(basic_results, dict) else None
            # Create dashboard (if method exists)
            dashboard = None
            if hasattr(self.visualizer, 'create_correlation_dashboard'):
                dashboard = self.visualizer.create_correlation_dashboard({
                    'correlation_matrix': main_corr,
                    'rolling': rolling_df,
                    'comparison': basic_results,
                    'advanced': advanced_results
                })
            result = {
                'fig_matrix': fig_matrix,
                'fig_rolling': fig_rolling,
                'fig_comparison': fig_comparison,
                'dashboard': dashboard
            }
            self.workflow_manager.complete_execution_step(step_id, message="Completed correlation visualizations", outputs=[str(result)])
            self.doc_engine.generate_html_report()
            return result

        builder = WorkflowBuilder(
            name="CorrelationsPipelineWorkflow",
            description="Agent-driven modular correlations pipeline workflow"
        )
        t1 = builder.add_task(
            name="BasicCorrelations",
            function=lambda data, workflow_context=None, **kwargs: task_basic_correlations(data),
            parameters={"data": data}
        ).current_tasks[0].task_id
        t2 = builder.add_task(
            name="RollingAnalysis",
            function=lambda data, workflow_context=None, **kwargs: task_rolling_analysis(data),
            parameters={"data": data}
        ).current_tasks[0].task_id
        t3 = builder.add_task(
            name="AdvancedAnalysis",
            function=lambda data, workflow_context=None, **kwargs: task_advanced_analysis(data),
            parameters={"data": data}
        ).current_tasks[0].task_id
        t4 = builder.add_task(
            name="Visualizations",
            function=lambda workflow_context=None, **kwargs: task_visualizations(kwargs[f'{t1}_output'], kwargs[f'{t2}_output'], kwargs[f'{t3}_output']),
            parameters={},
            dependencies=[t1, t2, t3]
        ).current_tasks[0].task_id
        workflow = builder.build()

        engine = WorkflowEngine()
        engine.register_workflow(workflow)
        result = engine.run(workflow)

        # Collect results from orchestrator result object
        self._results = {
            'basic_correlations': result.task_results[t1].output if t1 in result.task_results else None,
            'rolling_analysis': result.task_results[t2].output if t2 in result.task_results else None,
            'advanced_analysis': result.task_results[t3].output if t3 in result.task_results else None,
            'visualizations': result.task_results[t4].output if t4 in result.task_results else None
        }

        # Document workflow execution (summary step)
        self.workflow_manager.add_execution_step(
            step_id="correlations_workflow_completed",
            macro_process_name="Correlation Analysis",
            process_description="Correlations workflow completed",
            responsible_agent_or_tool="AgentCorrelationsPipeline",
            inputs_used=[str(data.shape)]
        )
        self.workflow_manager.start_execution_step("correlations_workflow_completed")
        self.workflow_manager.complete_execution_step(
            "correlations_workflow_completed",
            message="Correlations workflow completed",
            outputs=[str(self._results)]
        )
        self.doc_engine.generate_html_report()
        return self

    def get_report(self) -> str:
        """Return a formatted correlations pipeline report."""
        return f"Correlations Pipeline Report\n{'='*40}\nResults: {self._results}"
