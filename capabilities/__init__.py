"""
Capabilities Discovery Package

This package provides intelligent discovery, search, and recommendation capabilities
across all documented system resources including methods, functions, agents, APIs,
processes, metrics, and workflows.
"""

from .capability_indexer import CapabilityIndexer, CapabilityIndex
from .capability_searcher import CapabilitySearcher, SearchQuery, SearchResult
from .capability_recommender import CapabilityRecommender, RecommendationRequest, Recommendation
from .capability_analyzer import CapabilityAnalyzer, CapabilityGraph, DependencyMap
from .capability_manager import CapabilityManager
from .capability_validator import CapabilityValidator, ValidationResult

__all__ = [
    'CapabilityManager',
    'CapabilityIndexer',
    'CapabilityIndex', 
    'CapabilitySearcher',
    'SearchQuery',
    'SearchResult',
    'CapabilityRecommender',
    'RecommendationRequest',
    'Recommendation',
    'CapabilityAnalyzer',
    'CapabilityGraph',
    'DependencyMap',
    'CapabilityValidator',
    'ValidationResult'
]

__version__ = '1.0.0'
__author__ = 'Multi-Agent Scientist System'
