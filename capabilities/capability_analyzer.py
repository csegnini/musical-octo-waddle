"""
Capability Analyzer Module

This module provides analysis capabilities for understanding relationships
between capabilities, dependency mapping, and system optimization.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .capability_indexer import CapabilityIndex, CapabilityMetadata, CapabilityType

logger = logging.getLogger(__name__)


@dataclass
class CapabilityGraph:
    """Graph representation of capability relationships."""
    nodes: Dict[str, CapabilityMetadata] = field(default_factory=dict)
    edges: Dict[str, Set[str]] = field(default_factory=dict)
    clusters: Dict[str, Set[str]] = field(default_factory=dict)
    
    
@dataclass
class DependencyMap:
    """Dependency mapping for capabilities."""
    direct_dependencies: Dict[str, Set[str]] = field(default_factory=dict)
    transitive_dependencies: Dict[str, Set[str]] = field(default_factory=dict)
    circular_dependencies: List[List[str]] = field(default_factory=list)


class CapabilityAnalyzer:
    """
    Analyzer for capability relationships and dependencies.
    
    This class provides analysis capabilities for understanding the
    relationships between different system capabilities.
    """
    
    def __init__(self, capability_index: CapabilityIndex):
        """
        Initialize capability analyzer.
        
        Args:
            capability_index: Indexed capability data
        """
        self.index = capability_index
        logger.info("Initialized capability analyzer")
    
    def build_capability_graph(self) -> CapabilityGraph:
        """
        Build a graph representation of capabilities.
        
        Returns:
            Capability graph
        """
        graph = CapabilityGraph()
        
        # Add nodes
        for cap_id, capability in self.index.capabilities.items():
            graph.nodes[cap_id] = capability
            graph.edges[cap_id] = set()
        
        # Add edges based on dependencies
        for cap_id, dependencies in self.index.dependency_graph.items():
            graph.edges[cap_id] = dependencies.copy()
        
        logger.debug(f"Built capability graph with {len(graph.nodes)} nodes")
        return graph
    
    def analyze_dependencies(self) -> DependencyMap:
        """
        Analyze capability dependencies.
        
        Returns:
            Dependency analysis results
        """
        dep_map = DependencyMap()
        
        # Direct dependencies
        dep_map.direct_dependencies = {
            cap_id: deps.copy() 
            for cap_id, deps in self.index.dependency_graph.items()
        }
        
        # Calculate transitive dependencies
        dep_map.transitive_dependencies = self._calculate_transitive_dependencies()
        
        # Detect circular dependencies
        dep_map.circular_dependencies = self._detect_circular_dependencies()
        
        logger.debug(f"Analyzed dependencies for {len(dep_map.direct_dependencies)} capabilities")
        return dep_map
    
    def _calculate_transitive_dependencies(self) -> Dict[str, Set[str]]:
        """Calculate transitive dependencies for all capabilities."""
        transitive = {}
        
        for cap_id in self.index.capabilities:
            transitive[cap_id] = self._get_transitive_deps(cap_id, set())
        
        return transitive
    
    def _get_transitive_deps(self, cap_id: str, visited: Set[str]) -> Set[str]:
        """Get transitive dependencies for a capability."""
        if cap_id in visited:
            return set()
        
        visited.add(cap_id)
        deps = set()
        
        direct_deps = self.index.dependency_graph.get(cap_id, set())
        deps.update(direct_deps)
        
        for dep in direct_deps:
            deps.update(self._get_transitive_deps(dep, visited.copy()))
        
        return deps
    
    def _detect_circular_dependencies(self) -> List[List[str]]:
        """Detect circular dependencies in the capability graph."""
        circular = []
        visited = set()
        
        for cap_id in self.index.capabilities:
            if cap_id not in visited:
                cycle = self._find_cycle(cap_id, [], set())
                if cycle:
                    circular.append(cycle)
                    visited.update(cycle)
        
        return circular
    
    def _find_cycle(self, current: str, path: List[str], visited: Set[str]) -> Optional[List[str]]:
        """Find cycles starting from current capability."""
        if current in path:
            # Found cycle
            cycle_start = path.index(current)
            return path[cycle_start:] + [current]
        
        if current in visited:
            return None
        
        visited.add(current)
        path.append(current)
        
        for dep in self.index.dependency_graph.get(current, set()):
            cycle = self._find_cycle(dep, path.copy(), visited)
            if cycle:
                return cycle
        
        return None
    
    def get_capability_clusters(self) -> Dict[str, Set[str]]:
        """
        Identify clusters of related capabilities.
        
        Returns:
            Dictionary of cluster name to capability IDs
        """
        clusters = {}
        
        # Cluster by type
        for cap_type, cap_ids in self.index.type_index.items():
            clusters[f"type_{cap_type.value}"] = cap_ids.copy()
        
        # Cluster by category
        for category, cap_ids in self.index.category_index.items():
            if category and category.strip():
                cluster_name = f"category_{category.replace(' ', '_').lower()}"
                clusters[cluster_name] = cap_ids.copy()
        
        # Cluster by tags
        for tag, cap_ids in self.index.tag_index.items():
            if len(cap_ids) >= 3:  # Only create clusters with 3+ capabilities
                cluster_name = f"tag_{tag.replace(' ', '_').lower()}"
                clusters[cluster_name] = cap_ids.copy()
        
        return clusters
