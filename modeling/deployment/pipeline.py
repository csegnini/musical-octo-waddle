"""
DeploymentPipeline: Agent-driven, orchestrated, and documented deployment workflow
This pipeline modularly executes deployment tasks (API creation, model serving, monitoring, etc.),
using the workflow orchestrator and documentation engine. Highly adaptable for agent-driven workflows.
"""

from typing import Optional, Dict, Any, List
from src.workflows import WorkflowManager, DocumentationEngine

class DeploymentPipeline:
    """
    Main deployment pipeline for agent/workflow orchestration and documentation.
    Highly adaptable: accepts custom deployment configs, steps, and deployment types.
    """
    def __init__(self, deployment_config: Optional[Dict[str, Any]] = None):
        self.deployment_config = deployment_config or {}
        self._results = {}


    def run_deployment_workflow(self, workflow_manager: Optional[WorkflowManager] = None, doc_engine: Optional[DocumentationEngine] = None, agent_config: Optional[Dict[str, Any]] = None, deployment_steps: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Orchestrator-centric, agent-driven, and documented deployment workflow.
        Uses orchestrator Workflow, Task, WorkflowEngine, and always integrates WorkflowManager/DocumentationEngine.
        """
        from src.workflow_orchestrator.core import Workflow, Task, WorkflowEngine
        from src.workflow_orchestrator.utils import WorkflowBuilder
        workflow_manager = workflow_manager or WorkflowManager(user_query="deployment workflow")
        doc_engine = doc_engine or DocumentationEngine(workflow_manager)
        deployment_steps = deployment_steps or [
            {"type": "flask_api", "params": {}}
        ]

        # Define orchestrator tasks
        def make_task(step, i):
            deploy_type = step.get("type", "flask_api")
            params = step.get("params", {})
            def task_fn(workflow_context=None, **kwargs):
                step_id = f"deploy_{deploy_type}_{i}"
                workflow_manager.add_execution_step(
                    step_id=step_id,
                    macro_process_name="Deployment",
                    process_description=f"Run {deploy_type}",
                    responsible_agent_or_tool="DeploymentPipeline",
                    inputs_used=[str(params)]
                )
                workflow_manager.start_execution_step(step_id)
                # Only supporting flask_api for demo, can be extended
                if deploy_type == "flask_api":
                    result = f"Flask API created with params: {params}"
                else:
                    result = f"Deployment step {deploy_type} executed with params: {params}"
                workflow_manager.complete_execution_step(step_id, message=f"Completed {deploy_type}", outputs=[str(result)])
                doc_engine.generate_html_report()
                return result
            return Task(
                name=f"Deploy_{deploy_type}_{i}",
                function=task_fn,
                parameters={}
            )

        builder = WorkflowBuilder(
            name="DeploymentPipelineWorkflow",
            description="Agent-driven modular deployment pipeline workflow"
        )
        task_ids = []
        for i, step in enumerate(deployment_steps):
            task_obj = make_task(step, i)
            t = builder.add_task(
                name=task_obj.name,
                function=task_obj.function,
                parameters={}
            ).current_tasks[0].task_id
            task_ids.append(t)
        workflow = builder.build()

        engine = WorkflowEngine()
        engine.register_workflow(workflow)
        result = engine.run(workflow)

        # Collect results from orchestrator context
        context = workflow.context
        results = {tid: context.get(f'{tid}_output') for tid in task_ids}
        self._results = results

        # Document workflow execution (summary step)
        workflow_manager.add_execution_step(
            step_id="deploy_pipeline_completed",
            macro_process_name="Deployment",
            process_description="Deployment pipeline completed",
            responsible_agent_or_tool="DeploymentPipeline",
            inputs_used=[str(self.deployment_config)]
        )
        workflow_manager.start_execution_step("deploy_pipeline_completed")
        workflow_manager.complete_execution_step(
            "deploy_pipeline_completed",
            message="Deployment pipeline completed",
            outputs=[str(results)]
        )
        doc_engine.generate_html_report()
        return results
