"""
Workflow Management Package

This package provides comprehensive workflow orchestration, documentation, and state management
for multi-agent systems. It handles workflow lifecycle, data inventory, agent coordination,
and maintains detailed documentation throughout the execution process.
"""

from .workflow_manager import WorkflowManager
from .workflow_state import WorkflowState, WorkflowStatus, StageStatus
from .data_inventory import DataInventoryManager, DataReference
from .agent_coordinator import AgentCoordinator, WorkOrder
from .documentation_engine import DocumentationEngine
from .workflow_updater import WorkflowUpdater

__all__ = [
    'WorkflowManager',
    'WorkflowState', 
    'WorkflowStatus',
    'StageStatus',
    'DataInventoryManager',
    'DataReference',
    'AgentCoordinator',
    'WorkOrder',
    'DocumentationEngine',
    'WorkflowUpdater'
]

__version__ = '1.0.0'
__author__ = 'Multi-Agent System Team'
