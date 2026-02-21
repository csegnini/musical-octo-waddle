"""
Capability Indexer Module

This module handles the indexing and organization of all system capabilities
including methods, functions, agents, APIs, processes, metrics, and workflows.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class CapabilityType(Enum):
    """Types of capabilities in the system."""
    METHOD = "method"
    FUNCTION = "function"
    AGENT = "agent"
    API = "api"
    PROCESS = "process"
    METRIC = "metric"
    WORKFLOW = "workflow"


class CapabilityStatus(Enum):
    """Status of capabilities."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"
    MAINTENANCE = "maintenance"


@dataclass
class CapabilityMetadata:
    """Metadata for a capability."""
    id: str
    name: str
    description: str
    type: CapabilityType
    status: CapabilityStatus = CapabilityStatus.ACTIVE
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    source_file: str = ""
    last_updated: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    responsible_agents: List[str] = field(default_factory=list)
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    usage_count: int = 0
    success_rate: float = 0.0
    checksum: str = ""
    
    def __post_init__(self):
        """Calculate checksum after initialization."""
        self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate checksum for the capability."""
        content = f"{self.id}{self.name}{self.description}{self.type.value}"
        return hashlib.md5(content.encode()).hexdigest()


@dataclass
class CapabilityIndex:
    """Complete index of all system capabilities."""
    capabilities: Dict[str, CapabilityMetadata] = field(default_factory=dict)
    type_index: Dict[CapabilityType, Set[str]] = field(default_factory=dict)
    tag_index: Dict[str, Set[str]] = field(default_factory=dict)
    category_index: Dict[str, Set[str]] = field(default_factory=dict)
    dependency_graph: Dict[str, Set[str]] = field(default_factory=dict)
    reverse_dependency_graph: Dict[str, Set[str]] = field(default_factory=dict)
    agent_capabilities: Dict[str, Set[str]] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    total_capabilities: int = 0
    
    def __post_init__(self):
        """Initialize indexes."""
        for cap_type in CapabilityType:
            self.type_index[cap_type] = set()


class CapabilityIndexer:
    """
    Indexes all system capabilities from configuration files.
    
    This class scans and indexes methods, functions, agents, APIs, processes,
    metrics, and workflows from their respective configuration files.
    """
    
    def __init__(self, config_root: str = "config"):
        """
        Initialize capability indexer.
        
        Args:
            config_root: Root directory containing configuration files
        """
        self.config_root = Path(config_root)
        self.index = CapabilityIndex()
        
        # Configuration file mappings
        self.config_mappings = {
            CapabilityType.METHOD: [
                "methods/methods.json",
                "methods/methods_*.json"
            ],
            CapabilityType.FUNCTION: [
                "functions/functions.json", 
                "functions/functions_custom.json"
            ],
            CapabilityType.AGENT: [
                "agents/agents.json"
            ],
            CapabilityType.API: [
                "APIs/apis_inventory.json"
            ],
            CapabilityType.PROCESS: [
                "processes/process.json",
                "processes/custom_processes.json"
            ],
            CapabilityType.METRIC: [
                "metrics/metrics.json"
            ]
        }
        
        logger.info(f"Initialized capability indexer with config root: {self.config_root}")
    
    def build_full_index(self) -> CapabilityIndex:
        """
        Build complete capability index from all configuration files.
        
        Returns:
            Complete capability index
        """
        logger.info("Building full capability index...")
        
        # Clear existing index
        self.index = CapabilityIndex()
        
        # Index each capability type
        for cap_type, config_files in self.config_mappings.items():
            try:
                self._index_capability_type(cap_type, config_files)
            except Exception as e:
                logger.error(f"Error indexing {cap_type.value}: {e}")
        
        # Build derived indexes
        self._build_derived_indexes()
        
        # Update statistics
        self.index.total_capabilities = len(self.index.capabilities)
        self.index.last_updated = datetime.utcnow()
        
        logger.info(f"Built index with {self.index.total_capabilities} capabilities")
        return self.index
    
    def _index_capability_type(self, cap_type: CapabilityType, config_files: List[str]) -> None:
        """Index capabilities of a specific type."""
        logger.debug(f"Indexing {cap_type.value} capabilities...")
        
        for config_pattern in config_files:
            config_files_found = self._resolve_config_files(config_pattern)
            
            for config_file in config_files_found:
                try:
                    self._index_config_file(cap_type, config_file)
                except Exception as e:
                    logger.error(f"Error indexing {config_file}: {e}")
    
    def _resolve_config_files(self, pattern: str) -> List[Path]:
        """Resolve config file patterns to actual files."""
        if "*" in pattern:
            # Handle wildcard patterns
            base_dir = self.config_root / Path(pattern).parent
            if base_dir.exists():
                pattern_name = Path(pattern).name.replace("*", "")
                return [f for f in base_dir.glob(f"*{pattern_name}") if f.is_file()]
        else:
            # Handle direct file paths
            config_file = self.config_root / pattern
            if config_file.exists():
                return [config_file]
        
        return []
    
    def _normalize_dependencies(self, deps: Any) -> List[str]:
        """Normalize dependencies to list of strings."""
        if not deps:
            return []
        
        if isinstance(deps, str):
            return [deps]
        
        if isinstance(deps, list):
            normalized = []
            for dep in deps:
                if isinstance(dep, dict):
                    # Extract ID from dict if it has one
                    normalized.append(dep.get('id', str(dep)))
                elif isinstance(dep, str):
                    normalized.append(dep)
                else:
                    normalized.append(str(dep))
            return normalized
        
        return [str(deps)]
    
    def _normalize_tags(self, tags: Any) -> List[str]:
        """Normalize tags to list of strings."""
        if not tags:
            return []
        
        if isinstance(tags, str):
            return [tags]
        
        if isinstance(tags, list):
            return [str(tag) for tag in tags]
        
        return [str(tags)]
    
    def _index_config_file(self, cap_type: CapabilityType, config_file: Path) -> None:
        """Index capabilities from a configuration file."""
        logger.debug(f"Processing {config_file}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract capabilities based on type
            if cap_type == CapabilityType.METHOD:
                self._index_methods(data, str(config_file))
            elif cap_type == CapabilityType.FUNCTION:
                self._index_functions(data, str(config_file))
            elif cap_type == CapabilityType.AGENT:
                self._index_agents(data, str(config_file))
            elif cap_type == CapabilityType.API:
                self._index_apis(data, str(config_file))
            elif cap_type == CapabilityType.PROCESS:
                self._index_processes(data, str(config_file))
            elif cap_type == CapabilityType.METRIC:
                self._index_metrics(data, str(config_file))
                
        except Exception as e:
            logger.error(f"Error processing {config_file}: {e}")
    
    def _index_methods(self, data: Dict[str, Any], source_file: str) -> None:
        """Index methods from methods configuration."""
        methods = data.get("methods", [])
        
        for method in methods:
            try:
                capability = CapabilityMetadata(
                    id=method.get("id", ""),
                    name=method.get("name", ""),
                    description=method.get("description", ""),
                    type=CapabilityType.METHOD,
                    tags=self._normalize_tags(method.get("tags", [])),
                    categories=[method.get("module_area", ""), method.get("sub_category", "")],
                    source_file=source_file,
                    responsible_agents=method.get("responsible_agent_types", []),
                    inputs=method.get("inputs", []),
                    outputs=method.get("outputs", []),
                    parameters=method.get("parameters", {}),
                    dependencies=self._normalize_dependencies(method.get("dependencies", []))
                )
                
                self.index.capabilities[capability.id] = capability
                self.index.type_index[CapabilityType.METHOD].add(capability.id)
                
            except Exception as e:
                logger.error(f"Error indexing method {method.get('id', 'unknown')}: {e}")
    
    def _index_functions(self, data: Dict[str, Any], source_file: str) -> None:
        """Index functions from functions configuration."""
        functions = data.get("functions", [])
        
        for function in functions:
            try:
                capability = CapabilityMetadata(
                    id=function.get("id", ""),
                    name=function.get("name", ""),
                    description=function.get("description", ""),
                    type=CapabilityType.FUNCTION,
                    tags=self._normalize_tags(function.get("tags", [])),
                    categories=[function.get("module_area", ""), function.get("sub_category", "")],
                    source_file=source_file,
                    responsible_agents=function.get("responsible_agent_types", []),
                    inputs=function.get("inputs", []),
                    outputs=function.get("outputs", []),
                    dependencies=self._normalize_dependencies(function.get("dependencies", []))
                )
                
                self.index.capabilities[capability.id] = capability
                self.index.type_index[CapabilityType.FUNCTION].add(capability.id)
                
            except Exception as e:
                logger.error(f"Error indexing function {function.get('id', 'unknown')}: {e}")
    
    def _index_agents(self, data: Dict[str, Any], source_file: str) -> None:
        """Index agents from agents configuration."""
        agents = data.get("agents", [])
        
        for agent in agents:
            try:
                capability = CapabilityMetadata(
                    id=agent.get("agent_id", ""),
                    name=agent.get("name", ""),
                    description=agent.get("description", ""),
                    type=CapabilityType.AGENT,
                    tags=agent.get("capabilities", []),
                    categories=[agent.get("type", "")],
                    source_file=source_file,
                    status=CapabilityStatus.ACTIVE if agent.get("status") == "online" else CapabilityStatus.INACTIVE,
                    performance_metrics={
                        "current_load": agent.get("current_load_percentage", 0),
                        "errors_24h": agent.get("errors_in_last_24h", 0),
                        "max_concurrent": agent.get("configuration", {}).get("max_concurrent_workflows", 1)
                    }
                )
                
                self.index.capabilities[capability.id] = capability
                self.index.type_index[CapabilityType.AGENT].add(capability.id)
                
                # Build agent capabilities mapping
                agent_id = capability.id
                if agent_id not in self.index.agent_capabilities:
                    self.index.agent_capabilities[agent_id] = set()
                self.index.agent_capabilities[agent_id].update(capability.tags)
                
            except Exception as e:
                logger.error(f"Error indexing agent {agent.get('agent_id', 'unknown')}: {e}")
    
    def _index_apis(self, data: List[Dict[str, Any]], source_file: str) -> None:
        """Index APIs from APIs configuration."""
        if isinstance(data, dict):
            apis = data.get("apis", [])
        else:
            apis = data
        
        for api in apis:
            try:
                capability = CapabilityMetadata(
                    id=api.get("name", "").lower().replace(" ", "_"),
                    name=api.get("name", ""),
                    description=api.get("description", ""),
                    type=CapabilityType.API,
                    tags=[api.get("category", "")],
                    categories=[api.get("category", "")],
                    source_file=source_file,
                    status=CapabilityStatus.ACTIVE if api.get("status") == "Active" else CapabilityStatus.INACTIVE,
                    parameters={
                        "base_url": api.get("base_url", ""),
                        "authentication": api.get("authentication", {}),
                        "rate_limiting": api.get("rate_limiting", {}),
                        "endpoints": api.get("endpoints", [])
                    }
                )
                
                self.index.capabilities[capability.id] = capability
                self.index.type_index[CapabilityType.API].add(capability.id)
                
            except Exception as e:
                logger.error(f"Error indexing API {api.get('name', 'unknown')}: {e}")
    
    def _index_processes(self, data: Dict[str, Any], source_file: str) -> None:
        """Index processes from processes configuration."""
        processes = data.get("processes", [])
        
        for process in processes:
            try:
                capability = CapabilityMetadata(
                    id=process.get("id", ""),
                    name=process.get("name", ""),
                    description=process.get("description", ""),
                    type=CapabilityType.PROCESS,
                    tags=process.get("tags", []),
                    categories=[process.get("category", "")],
                    source_file=source_file,
                    dependencies=self._normalize_dependencies(process.get("dependencies", [])),
                    inputs=process.get("inputs", []),
                    outputs=process.get("outputs", [])
                )
                
                self.index.capabilities[capability.id] = capability
                self.index.type_index[CapabilityType.PROCESS].add(capability.id)
                
            except Exception as e:
                logger.error(f"Error indexing process {process.get('id', 'unknown')}: {e}")
    
    def _index_metrics(self, data: Dict[str, Any], source_file: str) -> None:
        """Index metrics from metrics configuration."""
        metrics = data.get("metrics", [])
        
        for metric in metrics:
            try:
                capability = CapabilityMetadata(
                    id=metric.get("id", ""),
                    name=metric.get("name", ""),
                    description=metric.get("description", ""),
                    type=CapabilityType.METRIC,
                    tags=metric.get("tags", []),
                    categories=[metric.get("category", "")],
                    source_file=source_file,
                    parameters=metric.get("parameters", {})
                )
                
                self.index.capabilities[capability.id] = capability
                self.index.type_index[CapabilityType.METRIC].add(capability.id)
                
            except Exception as e:
                logger.error(f"Error indexing metric {metric.get('id', 'unknown')}: {e}")
    
    def _build_derived_indexes(self) -> None:
        """Build derived indexes (tags, categories, dependencies)."""
        logger.debug("Building derived indexes...")
        
        # Build tag and category indexes
        for cap_id, capability in self.index.capabilities.items():
            # Tag index
            for tag in capability.tags:
                if tag and tag.strip():
                    tag = tag.strip().lower()
                    if tag not in self.index.tag_index:
                        self.index.tag_index[tag] = set()
                    self.index.tag_index[tag].add(cap_id)
            
            # Category index
            for category in capability.categories:
                if category and category.strip():
                    category = category.strip()
                    if category not in self.index.category_index:
                        self.index.category_index[category] = set()
                    self.index.category_index[category].add(cap_id)
            
            # Dependency graph - ensure dependencies are strings
            if cap_id not in self.index.dependency_graph:
                self.index.dependency_graph[cap_id] = set()
            
            # Handle dependencies that might be dicts or strings
            deps = []
            for dep in capability.dependencies:
                if isinstance(dep, dict):
                    # Extract ID from dict if it has one
                    deps.append(dep.get('id', str(dep)))
                elif isinstance(dep, str):
                    deps.append(dep)
                else:
                    deps.append(str(dep))
            
            self.index.dependency_graph[cap_id].update(deps)
            
            # Reverse dependency graph
            for dep in deps:
                if dep not in self.index.reverse_dependency_graph:
                    self.index.reverse_dependency_graph[dep] = set()
                self.index.reverse_dependency_graph[dep].add(cap_id)
    
    def update_capability_stats(self, capability_id: str, usage_count: Optional[int] = None, 
                              success_rate: Optional[float] = None) -> bool:
        """
        Update capability usage statistics.
        
        Args:
            capability_id: ID of the capability to update
            usage_count: New usage count
            success_rate: New success rate
            
        Returns:
            True if updated successfully
        """
        if capability_id not in self.index.capabilities:
            return False
        
        capability = self.index.capabilities[capability_id]
        
        if usage_count is not None:
            capability.usage_count = usage_count
        
        if success_rate is not None:
            capability.success_rate = success_rate
        
        logger.debug(f"Updated stats for {capability_id}")
        return True
    
    def get_capability_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the capability index.
        
        Returns:
            Summary statistics
        """
        summary = {
            "total_capabilities": self.index.total_capabilities,
            "last_updated": self.index.last_updated.isoformat(),
            "by_type": {
                cap_type.value: len(cap_ids) 
                for cap_type, cap_ids in self.index.type_index.items()
            },
            "top_tags": self._get_top_items(self.index.tag_index, 10),
            "top_categories": self._get_top_items(self.index.category_index, 10),
            "agents_count": len(self.index.agent_capabilities),
            "dependency_count": sum(len(deps) for deps in self.index.dependency_graph.values())
        }
        
        return summary
    
    def _get_top_items(self, index_dict: Dict[str, Set[str]], limit: int) -> List[Dict[str, Any]]:
        """Get top items by frequency."""
        items = [(item, len(cap_ids)) for item, cap_ids in index_dict.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        return [{"name": item, "count": count} for item, count in items[:limit]]
    
    def export_index(self, output_path: str) -> None:
        """
        Export the capability index to a JSON file.
        
        Args:
            output_path: Path to export the index
        """
        try:
            # Convert index to serializable format
            export_data = {
                "capabilities": {
                    cap_id: {
                        "id": cap.id,
                        "name": cap.name,
                        "description": cap.description,
                        "type": cap.type.value,
                        "status": cap.status.value,
                        "tags": cap.tags,
                        "categories": cap.categories,
                        "dependencies": cap.dependencies,
                        "source_file": cap.source_file,
                        "last_updated": cap.last_updated.isoformat(),
                        "version": cap.version,
                        "responsible_agents": cap.responsible_agents,
                        "inputs": cap.inputs,
                        "outputs": cap.outputs,
                        "parameters": cap.parameters,
                        "performance_metrics": cap.performance_metrics,
                        "usage_count": cap.usage_count,
                        "success_rate": cap.success_rate,
                        "checksum": cap.checksum
                    }
                    for cap_id, cap in self.index.capabilities.items()
                },
                "summary": self.get_capability_summary()
            }
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported capability index to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting index: {e}")
            raise
