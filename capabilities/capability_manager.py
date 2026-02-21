"""
Capability Manager Module

This module provides the main interface for the capabilities discovery package,
coordinating indexing, searching, and recommendation functionalities.
"""

import logging
import os
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path

from .capability_indexer import CapabilityIndexer, CapabilityIndex, CapabilityType, CapabilityStatus
from .capability_searcher import CapabilitySearcher, SearchQuery, SearchResult, SearchMode, SearchScope, SortBy
from .capability_recommender import (
    CapabilityRecommender, 
    RecommendationRequest, 
    Recommendation, 
    RecommendationType,
    DataCharacteristics,
    TaskContext
)

logger = logging.getLogger(__name__)


class CapabilityManager:
    """
    Main interface for the capabilities discovery system.
    
    This class coordinates all capability-related operations including indexing,
    searching, and recommendations. It provides a unified API for discovering
    and working with system capabilities.
    """
    
    def __init__(self, config_root: str = "config", cache_path: Optional[str] = None):
        """
        Initialize capability manager.
        
        Args:
            config_root: Root directory containing configuration files
            cache_path: Path to cache indexed capabilities
        """
        self.config_root = config_root
        self.cache_path = cache_path or os.path.join(config_root, "capabilities_cache.json")
        
        # Initialize components
        self.indexer = CapabilityIndexer(config_root)
        self.index: Optional[CapabilityIndex] = None
        self.searcher: Optional[CapabilitySearcher] = None
        self.recommender: Optional[CapabilityRecommender] = None
        
        # Build initial index
        self._initialize_system()
        
        logger.info(f"Initialized capability manager with {len(self.index.capabilities) if self.index else 0} capabilities")
    
    def _initialize_system(self) -> None:
        """Initialize the capability system."""
        try:
            # Try to load from cache first
            if os.path.exists(self.cache_path):
                try:
                    self._load_from_cache()
                    logger.info("Loaded capabilities from cache")
                except Exception as e:
                    logger.warning(f"Failed to load from cache: {e}")
                    self._rebuild_index()
            else:
                self._rebuild_index()
            
            # Initialize searcher and recommender
            if self.index:
                self.searcher = CapabilitySearcher(self.index)
                self.recommender = CapabilityRecommender(self.index, self.searcher)
            
        except Exception as e:
            logger.error(f"Failed to initialize capability system: {e}")
            raise
    
    def _rebuild_index(self) -> None:
        """Rebuild the capability index from configuration files."""
        logger.info("Building capability index from configuration files...")
        
        self.index = self.indexer.build_full_index()
        
        # Save to cache
        self._save_to_cache()
        
        logger.info(f"Built index with {self.index.total_capabilities} capabilities")
    
    def _load_from_cache(self) -> None:
        """Load capability index from cache."""
        with open(self.cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        # Reconstruct index from cache
        # This is a simplified version - in production, implement full deserialization
        self.index = CapabilityIndex()
        
        # Load basic statistics
        summary = cache_data.get("summary", {})
        self.index.total_capabilities = summary.get("total_capabilities", 0)
        
        if "last_updated" in summary:
            self.index.last_updated = datetime.fromisoformat(summary["last_updated"])
    
    def _save_to_cache(self) -> None:
        """Save capability index to cache."""
        if not self.index:
            return
        
        try:
            self.indexer.export_index(self.cache_path)
            logger.debug(f"Saved capabilities cache to {self.cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def refresh_index(self, force: bool = False) -> bool:
        """
        Refresh the capability index.
        
        Args:
            force: Force rebuild even if cache is recent
            
        Returns:
            True if index was refreshed
        """
        try:
            should_refresh = force
            
            if not force and self.index:
                # Check if cache is older than config files
                cache_time = self.index.last_updated
                
                # Check modification times of config files
                config_paths = [
                    Path(self.config_root) / "methods",
                    Path(self.config_root) / "functions", 
                    Path(self.config_root) / "agents",
                    Path(self.config_root) / "APIs",
                    Path(self.config_root) / "processes",
                    Path(self.config_root) / "metrics"
                ]
                
                for config_path in config_paths:
                    if config_path.exists():
                        for config_file in config_path.glob("*.json"):
                            file_time = datetime.fromtimestamp(config_file.stat().st_mtime)
                            if file_time > cache_time:
                                should_refresh = True
                                break
                    if should_refresh:
                        break
            
            if should_refresh:
                self._rebuild_index()
                
                # Reinitialize searcher and recommender
                if self.index:
                    self.searcher = CapabilitySearcher(self.index)
                    self.recommender = CapabilityRecommender(self.index, self.searcher)
                
                logger.info("Capability index refreshed")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to refresh index: {e}")
            return False
    
    def search_capabilities(
        self,
        query: str,
        capability_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        search_mode: str = "hybrid",
        max_results: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search for capabilities.
        
        Args:
            query: Search query
            capability_types: Filter by capability types
            tags: Filter by tags
            search_mode: Search mode ("exact", "fuzzy", "semantic", "hybrid")
            max_results: Maximum number of results
            
        Returns:
            List of search results
        """
        if not self.searcher:
            logger.error("Searcher not initialized")
            return []
        
        try:
            # Convert string types to enums
            mode_map = {
                "exact": SearchMode.EXACT,
                "fuzzy": SearchMode.FUZZY,
                "semantic": SearchMode.SEMANTIC,
                "hybrid": SearchMode.HYBRID
            }
            
            type_map = {
                "method": CapabilityType.METHOD,
                "function": CapabilityType.FUNCTION,
                "agent": CapabilityType.AGENT,
                "api": CapabilityType.API,
                "process": CapabilityType.PROCESS,
                "metric": CapabilityType.METRIC,
                "workflow": CapabilityType.WORKFLOW
            }
            
            search_query = SearchQuery(
                query=query,
                capability_types=[type_map[t] for t in (capability_types or []) if t in type_map],
                tags=tags,
                search_mode=mode_map.get(search_mode, SearchMode.HYBRID),
                max_results=max_results
            )
            
            results = self.searcher.search(search_query)
            
            # Convert to serializable format
            return [self._result_to_dict(result) for result in results]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_recommendations(
        self,
        query: str,
        data_types: Optional[List[str]] = None,
        task_type: Optional[str] = None,
        objectives: Optional[List[str]] = None,
        max_recommendations: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get capability recommendations.
        
        Args:
            query: Description of what you want to accomplish
            data_types: Types of data you're working with
            task_type: Type of task (classification, regression, etc.)
            objectives: List of objectives
            max_recommendations: Maximum number of recommendations
            
        Returns:
            List of recommendations
        """
        if not self.recommender:
            logger.error("Recommender not initialized")
            return []
        
        try:
            # Build recommendation request
            data_characteristics = None
            if data_types:
                data_characteristics = DataCharacteristics(data_types=data_types)
            
            task_context = None
            if task_type:
                task_context = TaskContext(task_type=task_type)
            
            request = RecommendationRequest(
                query=query,
                data_characteristics=data_characteristics,
                task_context=task_context,
                objectives=objectives or [],
                max_recommendations=max_recommendations
            )
            
            recommendations = self.recommender.get_recommendations(request)
            
            # Convert to serializable format
            return [self._recommendation_to_dict(rec) for rec in recommendations]
            
        except Exception as e:
            logger.error(f"Recommendation failed: {e}")
            return []
    
    def get_capability_details(self, capability_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific capability.
        
        Args:
            capability_id: ID of the capability
            
        Returns:
            Capability details or None if not found
        """
        if not self.index or capability_id not in self.index.capabilities:
            return None
        
        capability = self.index.capabilities[capability_id]
        
        return {
            "id": capability.id,
            "name": capability.name,
            "description": capability.description,
            "type": capability.type.value,
            "status": capability.status.value,
            "tags": capability.tags,
            "categories": capability.categories,
            "dependencies": capability.dependencies,
            "source_file": capability.source_file,
            "last_updated": capability.last_updated.isoformat(),
            "version": capability.version,
            "responsible_agents": capability.responsible_agents,
            "inputs": capability.inputs,
            "outputs": capability.outputs,
            "parameters": capability.parameters,
            "performance_metrics": capability.performance_metrics,
            "usage_count": capability.usage_count,
            "success_rate": capability.success_rate
        }
    
    def get_related_capabilities(self, capability_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get capabilities related to the given capability.
        
        Args:
            capability_id: ID of the capability
            limit: Maximum number of related capabilities
            
        Returns:
            List of related capabilities
        """
        if not self.searcher:
            return []
        
        try:
            related_results = self.searcher.get_related_capabilities(capability_id, limit)
            return [self._result_to_dict(result) for result in related_results]
        except Exception as e:
            logger.error(f"Failed to get related capabilities: {e}")
            return []
    
    def suggest_capability_names(self, partial_query: str, limit: int = 10) -> List[str]:
        """
        Get capability name suggestions based on partial query.
        
        Args:
            partial_query: Partial search query
            limit: Maximum number of suggestions
            
        Returns:
            List of suggested capability names
        """
        if not self.searcher:
            return []
        
        try:
            return self.searcher.suggest_capabilities(partial_query, limit)
        except Exception as e:
            logger.error(f"Failed to get suggestions: {e}")
            return []
    
    def get_capability_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the capability system.
        
        Returns:
            System statistics
        """
        if not self.index:
            return {}
        
        return self.indexer.get_capability_summary()
    
    def get_capabilities_by_type(self, capability_type: str) -> List[Dict[str, Any]]:
        """
        Get all capabilities of a specific type.
        
        Args:
            capability_type: Type of capabilities to retrieve
            
        Returns:
            List of capabilities
        """
        if not self.index:
            return []
        
        type_map = {
            "method": CapabilityType.METHOD,
            "function": CapabilityType.FUNCTION,
            "agent": CapabilityType.AGENT,
            "api": CapabilityType.API,
            "process": CapabilityType.PROCESS,
            "metric": CapabilityType.METRIC,
            "workflow": CapabilityType.WORKFLOW
        }
        
        cap_type = type_map.get(capability_type.lower())
        if not cap_type:
            return []
        
        capability_ids = self.index.type_index.get(cap_type, set())
        
        capabilities = []
        for cap_id in capability_ids:
            if cap_id in self.index.capabilities:
                cap_details = self.get_capability_details(cap_id)
                if cap_details:
                    capabilities.append(cap_details)
        
        return sorted(capabilities, key=lambda x: x["name"])
    
    def get_capabilities_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """
        Get all capabilities with a specific tag.
        
        Args:
            tag: Tag to filter by
            
        Returns:
            List of capabilities
        """
        if not self.index:
            return []
        
        capability_ids = self.index.tag_index.get(tag.lower(), set())
        
        capabilities = []
        for cap_id in capability_ids:
            if cap_id in self.index.capabilities:
                cap_details = self.get_capability_details(cap_id)
                if cap_details:
                    capabilities.append(cap_details)
        
        return sorted(capabilities, key=lambda x: x["name"])
    
    def update_capability_usage(self, capability_id: str, usage_count: int, success_rate: float) -> bool:
        """
        Update usage statistics for a capability.
        
        Args:
            capability_id: ID of the capability
            usage_count: New usage count
            success_rate: New success rate
            
        Returns:
            True if updated successfully
        """
        try:
            success = self.indexer.update_capability_stats(capability_id, usage_count, success_rate)
            
            if success:
                # Save updated cache
                self._save_to_cache()
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update capability usage: {e}")
            return False
    
    def export_capabilities(self, output_path: str, format: str = "json") -> bool:
        """
        Export capabilities to a file.
        
        Args:
            output_path: Path to export file
            format: Export format ("json", "csv")
            
        Returns:
            True if exported successfully
        """
        try:
            if format.lower() == "json":
                self.indexer.export_index(output_path)
                return True
            elif format.lower() == "csv":
                # Implement CSV export
                import csv
                
                with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                    if not self.index:
                        return False
                    
                    fieldnames = ['id', 'name', 'type', 'description', 'tags', 'categories', 'usage_count', 'success_rate']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    writer.writeheader()
                    for capability in self.index.capabilities.values():
                        writer.writerow({
                            'id': capability.id,
                            'name': capability.name,
                            'type': capability.type.value,
                            'description': capability.description,
                            'tags': ', '.join(capability.tags),
                            'categories': ', '.join(capability.categories),
                            'usage_count': capability.usage_count,
                            'success_rate': capability.success_rate
                        })
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to export capabilities: {e}")
            return False
    
    def _result_to_dict(self, result: SearchResult) -> Dict[str, Any]:
        """Convert search result to dictionary."""
        return {
            "capability": {
                "id": result.capability.id,
                "name": result.capability.name,
                "description": result.capability.description,
                "type": result.capability.type.value,
                "tags": result.capability.tags,
                "categories": result.capability.categories
            },
            "score": result.score,
            "match_details": result.match_details,
            "highlighted_fields": result.highlighted_fields
        }
    
    def _recommendation_to_dict(self, recommendation: Recommendation) -> Dict[str, Any]:
        """Convert recommendation to dictionary."""
        return {
            "capability": {
                "id": recommendation.capability.id,
                "name": recommendation.capability.name,
                "description": recommendation.capability.description,
                "type": recommendation.capability.type.value,
                "tags": recommendation.capability.tags,
                "categories": recommendation.capability.categories
            },
            "recommendation_type": recommendation.recommendation_type.value,
            "confidence": recommendation.confidence.value,
            "score": recommendation.score,
            "justification": recommendation.justification,
            "pros": recommendation.pros,
            "cons": recommendation.cons,
            "prerequisites": recommendation.prerequisites,
            "alternatives": recommendation.alternatives,
            "estimated_effort": recommendation.estimated_effort,
            "expected_impact": recommendation.expected_impact,
            "pipeline_position": recommendation.pipeline_position
        }
