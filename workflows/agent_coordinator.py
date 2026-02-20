"""
Agent Coordinator Module

This module provides agent coordination, work order management, and task distribution
for multi-agent workflow orchestration systems.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4
import json

logger = logging.getLogger(__name__)


class WorkOrderStatus(Enum):
    """Enumeration of work order statuses."""
    PENDING = "PENDING"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    RETRYING = "RETRYING"


class AgentStatus(Enum):
    """Enumeration of agent statuses."""
    AVAILABLE = "AVAILABLE"
    BUSY = "BUSY"
    OFFLINE = "OFFLINE"
    ERROR = "ERROR"
    MAINTENANCE = "MAINTENANCE"


@dataclass
class ToolConfig:
    """Configuration for tools assigned to work orders."""
    library: str
    methods: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    version: str = ""
    requirements: List[str] = field(default_factory=list)
    
    def add_method(self, method_name: str, parameters: Optional[Dict[str, Any]] = None) -> None:
        """Add a method to the tool configuration."""
        if method_name not in self.methods:
            self.methods.append(method_name)
        if parameters:
            self.parameters[method_name] = parameters
        logger.debug(f"Added method {method_name} to {self.library} tool config")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool config to dictionary."""
        return {
            'library': self.library,
            'methods': self.methods,
            'parameters': self.parameters,
            'version': self.version,
            'requirements': self.requirements
        }


@dataclass
class WorkOrder:
    """
    Represents a work order for agent execution.
    """
    order_id: str
    instructions: str
    macro_process: str = ""
    input_references: List[str] = field(default_factory=list)
    output_references_expected: List[str] = field(default_factory=list)
    assigned_tool_config: Optional[ToolConfig] = None
    log_path: str = ""
    assigned_agent: str = ""
    status: WorkOrderStatus = WorkOrderStatus.PENDING
    priority: int = 0  # Higher number = higher priority
    created_at: str = ""
    assigned_at: str = ""
    started_at: str = ""
    completed_at: str = ""
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 3600  # 1 hour default
    dependencies: List[str] = field(default_factory=list)  # Other order IDs
    metadata: Dict[str, Any] = field(default_factory=dict)
    result_data: Dict[str, Any] = field(default_factory=dict)
    error_details: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize computed fields after creation."""
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat() + "Z"
        if not self.order_id:
            self.order_id = str(uuid4())
    
    def assign_to_agent(self, agent_name: str) -> None:
        """Assign work order to an agent."""
        self.assigned_agent = agent_name
        self.assigned_at = datetime.utcnow().isoformat() + "Z"
        self.status = WorkOrderStatus.ASSIGNED
        logger.info(f"Work order {self.order_id} assigned to agent {agent_name}")
    
    def start_execution(self) -> None:
        """Mark work order as started."""
        self.started_at = datetime.utcnow().isoformat() + "Z"
        self.status = WorkOrderStatus.IN_PROGRESS
        logger.info(f"Work order {self.order_id} started execution")
    
    def complete_execution(self, result_data: Optional[Dict[str, Any]] = None) -> None:
        """Mark work order as completed."""
        self.completed_at = datetime.utcnow().isoformat() + "Z"
        self.status = WorkOrderStatus.COMPLETED
        if result_data:
            self.result_data.update(result_data)
        logger.info(f"Work order {self.order_id} completed successfully")
    
    def fail_execution(self, error_details: Dict[str, Any]) -> None:
        """Mark work order as failed."""
        self.status = WorkOrderStatus.FAILED
        self.error_details = error_details
        self.completed_at = datetime.utcnow().isoformat() + "Z"
        logger.error(f"Work order {self.order_id} failed: {error_details.get('message', 'Unknown error')}")
    
    def retry_execution(self) -> bool:
        """
        Attempt to retry execution if retries available.
        
        Returns:
            True if retry was initiated, False if max retries exceeded
        """
        if self.retry_count >= self.max_retries:
            logger.warning(f"Work order {self.order_id} exceeded max retries ({self.max_retries})")
            return False
        
        self.retry_count += 1
        self.status = WorkOrderStatus.RETRYING
        self.started_at = datetime.utcnow().isoformat() + "Z"
        self.error_details = None
        logger.info(f"Retrying work order {self.order_id} (attempt {self.retry_count + 1})")
        return True
    
    def cancel_execution(self, reason: str = "") -> None:
        """Cancel work order execution."""
        self.status = WorkOrderStatus.CANCELLED
        self.completed_at = datetime.utcnow().isoformat() + "Z"
        if reason:
            self.metadata['cancellation_reason'] = reason
        logger.info(f"Work order {self.order_id} cancelled: {reason}")
    
    def add_dependency(self, dependency_order_id: str) -> None:
        """Add a dependency on another work order."""
        if dependency_order_id not in self.dependencies:
            self.dependencies.append(dependency_order_id)
            logger.debug(f"Added dependency {dependency_order_id} to work order {self.order_id}")
    
    def get_execution_duration(self) -> Optional[float]:
        """Get execution duration in seconds."""
        if not self.started_at or not self.completed_at:
            return None
        
        start_time = datetime.fromisoformat(self.started_at.replace('Z', '+00:00'))
        end_time = datetime.fromisoformat(self.completed_at.replace('Z', '+00:00'))
        return (end_time - start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert work order to dictionary."""
        return {
            'order_id': self.order_id,
            'instructions': self.instructions,
            'macro_process': self.macro_process,
            'input_references': self.input_references,
            'output_references_expected': self.output_references_expected,
            'assigned_tool_config': self.assigned_tool_config.to_dict() if self.assigned_tool_config else None,
            'log_path': self.log_path,
            'assigned_agent': self.assigned_agent,
            'status': self.status.value,
            'priority': self.priority,
            'created_at': self.created_at,
            'assigned_at': self.assigned_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'timeout_seconds': self.timeout_seconds,
            'dependencies': self.dependencies,
            'metadata': self.metadata,
            'result_data': self.result_data,
            'error_details': self.error_details,
            'execution_duration': self.get_execution_duration()
        }


@dataclass
class Agent:
    """Represents an agent in the multi-agent system."""
    agent_id: str
    name: str
    capabilities: List[str] = field(default_factory=list)
    status: AgentStatus = AgentStatus.AVAILABLE
    current_work_order: Optional[str] = None
    total_orders_completed: int = 0
    total_orders_failed: int = 0
    average_execution_time: float = 0.0
    last_activity: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize computed fields after creation."""
        if not self.last_activity:
            self.last_activity = datetime.utcnow().isoformat() + "Z"
    
    def assign_work_order(self, work_order_id: str) -> None:
        """Assign a work order to this agent."""
        self.current_work_order = work_order_id
        self.status = AgentStatus.BUSY
        self.last_activity = datetime.utcnow().isoformat() + "Z"
        logger.info(f"Agent {self.name} assigned work order {work_order_id}")
    
    def complete_work_order(self, execution_time: float = 0.0) -> None:
        """Mark current work order as completed."""
        self.current_work_order = None
        self.status = AgentStatus.AVAILABLE
        self.total_orders_completed += 1
        self.last_activity = datetime.utcnow().isoformat() + "Z"
        
        # Update average execution time
        if execution_time > 0:
            total_time = self.average_execution_time * (self.total_orders_completed - 1) + execution_time
            self.average_execution_time = total_time / self.total_orders_completed
        
        logger.info(f"Agent {self.name} completed work order (total: {self.total_orders_completed})")
    
    def fail_work_order(self) -> None:
        """Mark current work order as failed."""
        self.current_work_order = None
        self.status = AgentStatus.AVAILABLE
        self.total_orders_failed += 1
        self.last_activity = datetime.utcnow().isoformat() + "Z"
        logger.warning(f"Agent {self.name} failed work order (total failures: {self.total_orders_failed})")
    
    def set_offline(self, reason: str = "") -> None:
        """Set agent status to offline."""
        self.status = AgentStatus.OFFLINE
        self.current_work_order = None
        if reason:
            self.metadata['offline_reason'] = reason
        logger.info(f"Agent {self.name} set to offline: {reason}")
    
    def set_online(self) -> None:
        """Set agent status to available."""
        self.status = AgentStatus.AVAILABLE
        self.last_activity = datetime.utcnow().isoformat() + "Z"
        if 'offline_reason' in self.metadata:
            del self.metadata['offline_reason']
        logger.info(f"Agent {self.name} is now online and available")
    
    def get_success_rate(self) -> float:
        """Calculate agent success rate."""
        total_orders = self.total_orders_completed + self.total_orders_failed
        if total_orders == 0:
            return 0.0
        return (self.total_orders_completed / total_orders) * 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary."""
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'capabilities': self.capabilities,
            'status': self.status.value,
            'current_work_order': self.current_work_order,
            'total_orders_completed': self.total_orders_completed,
            'total_orders_failed': self.total_orders_failed,
            'average_execution_time': self.average_execution_time,
            'success_rate': self.get_success_rate(),
            'last_activity': self.last_activity,
            'metadata': self.metadata
        }


class AgentCoordinator:
    """
    Agent coordination and work order management system.
    
    This class manages agents, work orders, task distribution, and provides
    comprehensive coordination capabilities for multi-agent workflows.
    """
    
    def __init__(self, workflow_id: str):
        """
        Initialize agent coordinator.
        
        Args:
            workflow_id: Unique identifier for the workflow
        """
        self.workflow_id = workflow_id
        self.agents: Dict[str, Agent] = {}
        self.work_orders: Dict[str, WorkOrder] = {}
        self.completed_orders: List[str] = []
        self.failed_orders: List[str] = []
        self.creation_timestamp = datetime.utcnow().isoformat() + "Z"
        
        logger.info(f"Initialized agent coordinator for workflow {workflow_id}")
    
    def register_agent(
        self, 
        agent_id: str, 
        name: str, 
        capabilities: Optional[List[str]] = None
    ) -> Agent:
        """
        Register a new agent.
        
        Args:
            agent_id: Unique agent identifier
            name: Human-readable agent name
            capabilities: List of agent capabilities
            
        Returns:
            Created Agent object
        """
        agent = Agent(
            agent_id=agent_id,
            name=name,
            capabilities=capabilities or []
        )
        self.agents[agent_id] = agent
        logger.info(f"Registered agent {name} ({agent_id}) with capabilities: {capabilities}")
        return agent
    
    def create_work_order(
        self,
        instructions: str,
        macro_process: str = "",
        input_references: Optional[List[str]] = None,
        output_references_expected: Optional[List[str]] = None,
        tool_config: Optional[ToolConfig] = None,
        priority: int = 0,
        timeout_seconds: int = 3600
    ) -> WorkOrder:
        """
        Create a new work order.
        
        Args:
            instructions: Detailed instructions for the work order
            macro_process: Name of the macro process
            input_references: List of input data references
            output_references_expected: List of expected output references
            tool_config: Configuration for tools to be used
            priority: Priority level (higher = more priority)
            timeout_seconds: Timeout for execution
            
        Returns:
            Created WorkOrder object
        """
        work_order = WorkOrder(
            order_id=str(uuid4()),
            instructions=instructions,
            macro_process=macro_process,
            input_references=input_references or [],
            output_references_expected=output_references_expected or [],
            assigned_tool_config=tool_config,
            priority=priority,
            timeout_seconds=timeout_seconds
        )
        
        self.work_orders[work_order.order_id] = work_order
        logger.info(f"Created work order {work_order.order_id}: {macro_process}")
        return work_order
    
    def assign_work_order(self, order_id: str, agent_id: str) -> bool:
        """
        Assign a work order to a specific agent.
        
        Args:
            order_id: Work order ID
            agent_id: Agent ID
            
        Returns:
            True if assignment successful, False otherwise
        """
        if order_id not in self.work_orders:
            logger.error(f"Work order {order_id} not found")
            return False
        
        if agent_id not in self.agents:
            logger.error(f"Agent {agent_id} not found")
            return False
        
        agent = self.agents[agent_id]
        work_order = self.work_orders[order_id]
        
        if agent.status != AgentStatus.AVAILABLE:
            logger.warning(f"Agent {agent_id} is not available (status: {agent.status.value})")
            return False
        
        # Check dependencies
        if not self._check_dependencies_satisfied(work_order):
            logger.info(f"Dependencies not satisfied for work order {order_id}")
            return False
        
        # Assign the work order
        work_order.assign_to_agent(agent.agent_id)
        agent.assign_work_order(order_id)
        
        return True
    
    def auto_assign_work_orders(self, work_order_requirements_key: str = 'requirements') -> int:
        """
        Automatically assign pending work orders to available agents.
        
        Returns:
            Number of work orders assigned
        """
        assigned_count = 0
        
        # Get pending work orders sorted by priority
        pending_orders = [
            wo for wo in self.work_orders.values() 
            if wo.status == WorkOrderStatus.PENDING
        ]
        pending_orders.sort(key=lambda x: (-x.priority, x.created_at))
        
        for work_order in pending_orders:
            # Check dependencies
            if not self._check_dependencies_satisfied(work_order):
                continue
            
            # Get required capabilities from the work order's metadata
            required_capabilities = set(work_order.metadata.get(work_order_requirements_key, []))
            
            # Find available agents that have ALL required capabilities
            suitable_agents = [
                agent for agent in self.agents.values()
                if (agent.status == AgentStatus.AVAILABLE and
                    required_capabilities.issubset(set(agent.capabilities)))
            ]
            
            if not suitable_agents:
                logger.debug(f"No available agents with capabilities {required_capabilities} for work order {work_order.order_id}")
                continue
            
            # Find best agent for this work order
            best_agent = self._find_best_agent(work_order, suitable_agents)
            if best_agent:
                if self.assign_work_order(work_order.order_id, best_agent.agent_id):
                    assigned_count += 1
                    available_agents.remove(best_agent)
        
        logger.info(f"Auto-assigned {assigned_count} work orders")
        return assigned_count
    
    def _check_dependencies_satisfied(self, work_order: WorkOrder) -> bool:
        """Check if all dependencies for a work order are satisfied."""
        for dep_id in work_order.dependencies:
            if dep_id in self.work_orders:
                dep_order = self.work_orders[dep_id]
                if dep_order.status != WorkOrderStatus.COMPLETED:
                    return False
            else:
                logger.warning(f"Dependency {dep_id} not found for work order {work_order.order_id}")
                return False
        return True
    
    def _find_best_agent(self, work_order: WorkOrder, available_agents: List[Agent]) -> Optional[Agent]:
        """Find the best agent for a work order based on capabilities and performance."""
        if not available_agents:
            return None
        
        scored_agents = []
        
        for agent in available_agents:
            score = 0
            
            # Score based on success rate
            score += agent.get_success_rate() * 0.4
            
            # Score based on experience (completed orders)
            if agent.total_orders_completed > 0:
                score += min(agent.total_orders_completed / 10, 10) * 0.3
            
            # Score based on average execution time (lower is better)
            if agent.average_execution_time > 0:
                time_score = max(0, 10 - (agent.average_execution_time / 60))  # Normalize to minutes
                score += time_score * 0.3
            
            scored_agents.append((agent, score))
        
        # Sort by score (descending) and return best agent
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        return scored_agents[0][0]
    
    def start_work_order(self, order_id: str) -> bool:
        """
        Start execution of a work order.
        
        Args:
            order_id: Work order ID
            
        Returns:
            True if started successfully, False otherwise
        """
        if order_id not in self.work_orders:
            logger.error(f"Work order {order_id} not found")
            return False
        
        work_order = self.work_orders[order_id]
        if work_order.status != WorkOrderStatus.ASSIGNED:
            logger.warning(f"Work order {order_id} is not in ASSIGNED status")
            return False
        
        work_order.start_execution()
        return True
    
    def complete_work_order(self, order_id: str, result_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Complete a work order.
        
        Args:
            order_id: Work order ID
            result_data: Optional result data
            
        Returns:
            True if completed successfully, False otherwise
        """
        if order_id not in self.work_orders:
            logger.error(f"Work order {order_id} not found")
            return False
        
        work_order = self.work_orders[order_id]
        agent_id = work_order.assigned_agent
        
        # Complete the work order
        work_order.complete_execution(result_data)
        self.completed_orders.append(order_id)
        
        # Update agent status
        agent = self.agents.get(agent_id)
        if agent:
            execution_time = work_order.get_execution_duration() or 0.0
            agent.complete_work_order(execution_time)
        
        return True
    
    def fail_work_order(self, order_id: str, error_details: Dict[str, Any]) -> bool:
        """
        Fail a work order.
        
        Args:
            order_id: Work order ID
            error_details: Error details
            
        Returns:
            True if failed successfully, False otherwise
        """
        if order_id not in self.work_orders:
            logger.error(f"Work order {order_id} not found")
            return False
        
        work_order = self.work_orders[order_id]
        agent_id = work_order.assigned_agent
        
        # Fail the work order
        work_order.fail_execution(error_details)
        self.failed_orders.append(order_id)
        
        # Update agent status
        agent = self.agents.get(agent_id)
        if agent:
            agent.fail_work_order()
        
        return True
    
    def retry_work_order(self, order_id: str) -> bool:
        """
        Retry a failed work order.
        
        Args:
            order_id: Work order ID
            
        Returns:
            True if retry initiated, False otherwise
        """
        if order_id not in self.work_orders:
            logger.error(f"Work order {order_id} not found")
            return False
        
        work_order = self.work_orders[order_id]
        if work_order.status != WorkOrderStatus.FAILED:
            logger.warning(f"Work order {order_id} is not in FAILED status")
            return False
        
        if work_order.retry_execution():
            # Remove from failed orders list
            if order_id in self.failed_orders:
                self.failed_orders.remove(order_id)
            return True
        
        return False
    
    def get_coordinator_statistics(self) -> Dict[str, Any]:
        """Get comprehensive coordinator statistics."""
        total_orders = len(self.work_orders)
        completed_orders = len(self.completed_orders)
        failed_orders = len(self.failed_orders)
        pending_orders = len([wo for wo in self.work_orders.values() if wo.status == WorkOrderStatus.PENDING])
        in_progress_orders = len([wo for wo in self.work_orders.values() if wo.status == WorkOrderStatus.IN_PROGRESS])
        
        agent_stats = {}
        for agent_id, agent in self.agents.items():
            agent_stats[agent_id] = agent.to_dict()
        
        return {
            'workflow_id': self.workflow_id,
            'creation_timestamp': self.creation_timestamp,
            'total_agents': len(self.agents),
            'available_agents': len([a for a in self.agents.values() if a.status == AgentStatus.AVAILABLE]),
            'busy_agents': len([a for a in self.agents.values() if a.status == AgentStatus.BUSY]),
            'total_work_orders': total_orders,
            'completed_orders': completed_orders,
            'failed_orders': failed_orders,
            'pending_orders': pending_orders,
            'in_progress_orders': in_progress_orders,
            'success_rate': (completed_orders / total_orders * 100) if total_orders > 0 else 0,
            'agent_statistics': agent_stats
        }
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export coordinator state to dictionary."""
        return {
            'workflow_id': self.workflow_id,
            'creation_timestamp': self.creation_timestamp,
            'agents': {agent_id: agent.to_dict() for agent_id, agent in self.agents.items()},
            'work_orders': {order_id: order.to_dict() for order_id, order in self.work_orders.items()},
            'completed_orders': self.completed_orders,
            'failed_orders': self.failed_orders,
            'statistics': self.get_coordinator_statistics()
        }
