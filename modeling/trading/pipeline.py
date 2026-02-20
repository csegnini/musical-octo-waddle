"""
AgentTradingPipeline: Highly Adaptable, Agent-Driven, Workflow-Orchestrated, and Documented Trading Pipeline

This pipeline enables modular or full execution of trading tasks by an agent, using workflow_orchestrator and workflows modules. The agent can assemble the workflow from config/capabilities and use the pipelines module in combination with workflow modules. Designed for maximum adaptability and modularity, following the orchestrator-centric pattern.
"""

import pandas as pd
from typing import Optional, Dict, Any
from src.trading.strategies import create_strategy, StrategyConfig
from src.trading.signals import AdvancedSignalGenerator, SignalConfig
from src.trading.portfolio import CorrelationPortfolioManager, PortfolioConfig
from src.trading.risk_management import AdvancedRiskManager, RiskConfig

class AgentTradingPipeline:
    """
    Highly adaptable trading pipeline for agent-driven, workflow-orchestrated, and documented execution.
    Enables modular or full trading tasks by an agent, using workflow_orchestrator and workflows modules.
    The agent can assemble the workflow from config/capabilities and use the pipelines module in combination with workflow modules.
    """
    def __init__(self,
                 initial_capital: float = 100000.0,
                 strategy_type: str = 'pairs',
                 strategy_config: Optional[StrategyConfig] = None,
                 signal_config: Optional[SignalConfig] = None,
                 portfolio_config: Optional[PortfolioConfig] = None,
                 risk_config: Optional[RiskConfig] = None,
                 workflow_manager=None,
                 doc_engine=None,
                 agent_config: Optional[Dict[str, Any]] = None):
        self.strategy = create_strategy(strategy_type, strategy_config)
        self.signal_generator = AdvancedSignalGenerator(signal_config)
        self.portfolio_manager = CorrelationPortfolioManager(initial_capital, portfolio_config)
        self.risk_manager = AdvancedRiskManager(risk_config)
        self.workflow_manager = workflow_manager
        self.doc_engine = doc_engine
        self.agent_config = agent_config or {}
        self._results = {}

    def run_workflow(self, data: pd.DataFrame, **kwargs):
        """
        Agent-driven, orchestrator-based, and documented execution of trading pipeline.
        Uses orchestrator Workflow, Task, WorkflowEngine, and always integrates WorkflowManager/DocumentationEngine.
        Legacy direct logic is deprecated but preserved for reference.
        Args:
            data: Input price data
            kwargs: Additional arguments for pipeline steps
        Returns:
            self: Fitted pipeline
        """
        # Always use WorkflowManager and DocumentationEngine
        if self.workflow_manager is None:
            from src.workflows.workflow_manager import WorkflowManager
            self.workflow_manager = WorkflowManager(user_query="trading workflow")
        if self.doc_engine is None:
            from src.workflows.documentation_engine import DocumentationEngine
            self.doc_engine = DocumentationEngine(self.workflow_manager)

        from src.workflow_orchestrator.core import Workflow, Task, WorkflowEngine
        from src.workflow_orchestrator.utils import WorkflowBuilder

        def task_generate_signals(data):
            step_id = "generate_signals"
            self.workflow_manager.add_execution_step(
                step_id=step_id,
                macro_process_name="Trading",
                process_description="Generate trading signals",
                responsible_agent_or_tool="AgentTradingPipeline",
                inputs_used=[str(data.shape)]
            )
            self.workflow_manager.start_execution_step(step_id)
            signals = self.strategy.generate_signals(data)
            self.workflow_manager.complete_execution_step(step_id, message="Generated trading signals", outputs=[str(signals)])
            self.doc_engine.generate_html_report()
            return signals

        def task_update_positions(signals, data):
            step_id = "update_positions"
            self.workflow_manager.add_execution_step(
                step_id=step_id,
                macro_process_name="Trading",
                process_description="Update positions based on signals",
                responsible_agent_or_tool="AgentTradingPipeline",
                inputs_used=[str(signals)]
            )
            self.workflow_manager.start_execution_step(step_id)
            current_prices = data.iloc[-1] if not data.empty else pd.Series()
            self.strategy.update_positions(signals, current_prices)
            self.workflow_manager.complete_execution_step(step_id, message="Updated positions", outputs=[str(self.strategy.positions)])
            self.doc_engine.generate_html_report()
            return self.strategy.positions

        def task_portfolio_management(positions, data):
            step_id = "portfolio_management"
            self.workflow_manager.add_execution_step(
                step_id=step_id,
                macro_process_name="Portfolio",
                process_description="Manage portfolio using signals and positions",
                responsible_agent_or_tool="AgentTradingPipeline",
                inputs_used=[str(positions)]
            )
            self.workflow_manager.start_execution_step(step_id)
            self.portfolio_manager.update_correlation_matrix(data)
            self.workflow_manager.complete_execution_step(step_id, message="Portfolio management updated", outputs=[str(self.portfolio_manager.correlation_matrix)])
            self.doc_engine.generate_html_report()
            return self.portfolio_manager.correlation_matrix

        def task_risk_management(correlation_matrix, positions):
            step_id = "risk_management"
            self.workflow_manager.add_execution_step(
                step_id=step_id,
                macro_process_name="Risk",
                process_description="Run risk management checks",
                responsible_agent_or_tool="AgentTradingPipeline",
                inputs_used=[str(correlation_matrix)]
            )
            self.workflow_manager.start_execution_step(step_id)
            portfolio_weights = {k: v['position_size'] for k, v in positions.items()} if positions else {}
            risk_metrics = self.risk_manager.calculate_correlation_risk_metrics(correlation_matrix, portfolio_weights)
            self.workflow_manager.complete_execution_step(step_id, message="Risk management completed", outputs=[str(risk_metrics)])
            self.doc_engine.generate_html_report()
            return risk_metrics

        builder = WorkflowBuilder(
            name="TradingPipelineWorkflow",
            description="Agent-driven modular trading pipeline workflow"
        )
        t1 = builder.add_task(
            name="GenerateSignals",
            function=lambda data, **kwargs: task_generate_signals(data),
            parameters={"data": data}
        ).current_tasks[0].task_id
        t2 = builder.add_task(
            name="UpdatePositions",
            function=lambda **kwargs: task_update_positions(kwargs[f'{t1}_output'], data),
            parameters={"data": data},
            dependencies=[t1]
        ).current_tasks[0].task_id
        t3 = builder.add_task(
            name="PortfolioManagement",
            function=lambda **kwargs: task_portfolio_management(kwargs[f'{t2}_output'], data),
            parameters={"data": data},
            dependencies=[t2]
        ).current_tasks[0].task_id
        t4 = builder.add_task(
            name="RiskManagement",
            function=lambda **kwargs: task_risk_management(kwargs.get(f'{t3}_output'), kwargs.get(f'{t2}_output')),
            parameters={},
            dependencies=[t3]
        ).current_tasks[0].task_id
        workflow = builder.build()

        engine = WorkflowEngine()
        engine.register_workflow(workflow)
        result = engine.run(workflow)

        # Collect results from orchestrator context
        context = workflow.context
        self._results = {
            'signals': context.get('GenerateSignals'),
            'positions': context.get('UpdatePositions'),
            'portfolio_correlation_matrix': context.get('PortfolioManagement'),
            'risk_metrics': context.get('RiskManagement')
        }

        # Document workflow execution (summary step)
        self.workflow_manager.add_execution_step(
            step_id="trading_workflow_completed",
            macro_process_name="Trading",
            process_description="Trading workflow completed",
            responsible_agent_or_tool="AgentTradingPipeline",
            inputs_used=[str(data.shape)]
        )
        self.workflow_manager.start_execution_step("trading_workflow_completed")
        self.workflow_manager.complete_execution_step(
            "trading_workflow_completed",
            message="Trading workflow completed",
            outputs=[str(self._results)]
        )
        self.doc_engine.generate_html_report()
        return self

    def get_report(self) -> str:
        """Return a formatted trading pipeline report."""
        return f"Trading Pipeline Report\n{'='*40}\nResults: {self._results}"
