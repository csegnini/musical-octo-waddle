"""
Capability Recommender Module

This module provides intelligent recommendations for capabilities based on context,
objectives, data characteristics, and historical patterns.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import math

from .capability_indexer import CapabilityIndex, CapabilityMetadata, CapabilityType, CapabilityStatus
from .capability_searcher import CapabilitySearcher, SearchQuery, SearchResult, SearchMode

logger = logging.getLogger(__name__)


class RecommendationType(Enum):
    """Types of recommendations."""
    TASK_BASED = "task_based"
    DATA_DRIVEN = "data_driven"
    OBJECTIVE_ORIENTED = "objective_oriented"
    PIPELINE_COMPLETION = "pipeline_completion"
    OPTIMIZATION = "optimization"
    ALTERNATIVE = "alternative"


class ConfidenceLevel(Enum):
    """Confidence levels for recommendations."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class DataCharacteristics:
    """Characteristics of input data for recommendations."""
    data_types: List[str] = field(default_factory=list)  # ["csv", "json", "parquet"]
    data_size: str = "medium"  # "small", "medium", "large", "xlarge"
    data_quality: str = "unknown"  # "high", "medium", "low", "unknown"
    missing_values: bool = False
    categorical_features: bool = False
    numerical_features: bool = False
    time_series: bool = False
    text_data: bool = False
    image_data: bool = False
    structured: bool = True
    feature_count: Optional[int] = None
    record_count: Optional[int] = None
    has_target_variable: bool = False
    is_labeled: bool = False


@dataclass
class TaskContext:
    """Context information for the task."""
    task_type: str = ""  # "classification", "regression", "clustering", "preprocessing"
    domain: str = ""  # "finance", "healthcare", "retail", etc.
    complexity: str = "medium"  # "low", "medium", "high"
    performance_requirements: List[str] = field(default_factory=list)  # ["speed", "accuracy", "interpretability"]
    constraints: List[str] = field(default_factory=list)  # ["memory", "time", "budget"]
    user_experience: str = "intermediate"  # "beginner", "intermediate", "expert"
    
    
@dataclass
class RecommendationRequest:
    """Request for capability recommendations."""
    query: str
    recommendation_type: RecommendationType = RecommendationType.TASK_BASED
    data_characteristics: Optional[DataCharacteristics] = None
    task_context: Optional[TaskContext] = None
    current_capabilities: List[str] = field(default_factory=list)
    objectives: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    preference_weights: Dict[str, float] = field(default_factory=dict)
    max_recommendations: int = 10
    include_alternatives: bool = True
    include_pipeline_suggestions: bool = True


@dataclass
class Recommendation:
    """Capability recommendation with justification."""
    capability: CapabilityMetadata
    recommendation_type: RecommendationType
    confidence: ConfidenceLevel
    score: float
    justification: str
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)
    usage_examples: List[str] = field(default_factory=list)
    estimated_effort: str = "medium"  # "low", "medium", "high"
    expected_impact: str = "medium"  # "low", "medium", "high"
    pipeline_position: Optional[str] = None  # "preprocessing", "analysis", "postprocessing"
    
    def __str__(self) -> str:
        return f"{self.capability.name} ({self.confidence.value} confidence) - {self.justification}"


class CapabilityRecommender:
    """
    Intelligent capability recommendation engine.
    
    Analyzes requirements, data characteristics, and objectives to suggest
    the most appropriate capabilities for a given task.
    """
    
    def __init__(self, capability_index: CapabilityIndex, searcher: CapabilitySearcher):
        """
        Initialize capability recommender.
        
        Args:
            capability_index: Indexed capability data
            searcher: Capability searcher instance
        """
        self.index = capability_index
        self.searcher = searcher
        
        # Build recommendation knowledge base
        self._build_recommendation_rules()
        self._build_compatibility_matrix()
        
        logger.info("Initialized capability recommender")
    
    def _build_recommendation_rules(self) -> None:
        """Build rules for capability recommendations."""
        # Data type to capability mapping
        self.data_type_rules = {
            "csv": ["data_cleaning", "data_transformation", "statistical_analysis"],
            "json": ["data_parsing", "data_flattening", "schema_validation"],
            "time_series": ["time_series_analysis", "forecasting", "seasonality_detection"],
            "text": ["text_preprocessing", "nlp", "sentiment_analysis"],
            "image": ["image_preprocessing", "computer_vision", "feature_extraction"]
        }
        
        # Task type to capability mapping
        self.task_type_rules = {
            "classification": ["feature_selection", "model_training", "evaluation"],
            "regression": ["feature_engineering", "model_fitting", "prediction"],
            "clustering": ["dimensionality_reduction", "clustering_algorithms", "visualization"],
            "preprocessing": ["data_cleaning", "normalization", "feature_engineering"],
            "analysis": ["statistical_analysis", "correlation_analysis", "visualization"]
        }
        
        # Quality to preprocessing mapping
        self.quality_rules = {
            "low": ["data_cleaning", "outlier_detection", "missing_value_imputation"],
            "medium": ["data_validation", "basic_cleaning"],
            "high": ["data_validation"]
        }
        
        # Performance requirements mapping
        self.performance_rules = {
            "speed": ["optimized_algorithms", "parallel_processing", "caching"],
            "accuracy": ["ensemble_methods", "hyperparameter_tuning", "cross_validation"],
            "interpretability": ["linear_models", "decision_trees", "feature_importance"]
        }
    
    def _build_compatibility_matrix(self) -> None:
        """Build compatibility matrix between capabilities."""
        self.compatibility_matrix = {}
        
        # Build based on typical ML pipelines
        pipeline_sequences = [
            ["data_loading", "data_cleaning", "feature_engineering", "model_training", "evaluation"],
            ["data_extraction", "preprocessing", "analysis", "visualization"],
            ["data_collection", "validation", "transformation", "modeling", "deployment"]
        ]
        
        for sequence in pipeline_sequences:
            for i, current in enumerate(sequence):
                if current not in self.compatibility_matrix:
                    self.compatibility_matrix[current] = {"follows": [], "precedes": []}
                
                if i > 0:
                    self.compatibility_matrix[current]["follows"].append(sequence[i-1])
                if i < len(sequence) - 1:
                    self.compatibility_matrix[current]["precedes"].append(sequence[i+1])
    
    def get_recommendations(self, request: RecommendationRequest) -> List[Recommendation]:
        """
        Get capability recommendations based on request.
        
        Args:
            request: Recommendation request with context
            
        Returns:
            List of ranked recommendations
        """
        logger.debug(f"Getting recommendations for: '{request.query}'")
        
        recommendations = []
        
        # Get base recommendations from different strategies
        if request.recommendation_type == RecommendationType.TASK_BASED:
            recommendations.extend(self._get_task_based_recommendations(request))
        elif request.recommendation_type == RecommendationType.DATA_DRIVEN:
            recommendations.extend(self._get_data_driven_recommendations(request))
        elif request.recommendation_type == RecommendationType.OBJECTIVE_ORIENTED:
            recommendations.extend(self._get_objective_oriented_recommendations(request))
        elif request.recommendation_type == RecommendationType.PIPELINE_COMPLETION:
            recommendations.extend(self._get_pipeline_completion_recommendations(request))
        else:
            # Hybrid approach - combine multiple strategies
            recommendations.extend(self._get_hybrid_recommendations(request))
        
        # Add alternatives if requested
        if request.include_alternatives:
            recommendations.extend(self._get_alternative_recommendations(recommendations, request))
        
        # Add pipeline suggestions if requested
        if request.include_pipeline_suggestions:
            recommendations.extend(self._get_pipeline_suggestions(recommendations, request))
        
        # Remove duplicates and score
        recommendations = self._deduplicate_recommendations(recommendations)
        recommendations = self._score_recommendations(recommendations, request)
        
        # Sort by score and limit results
        recommendations.sort(key=lambda r: r.score, reverse=True)
        
        if request.max_recommendations > 0:
            recommendations = recommendations[:request.max_recommendations]
        
        logger.debug(f"Generated {len(recommendations)} recommendations")
        return recommendations
    
    def _get_task_based_recommendations(self, request: RecommendationRequest) -> List[Recommendation]:
        """Get recommendations based on task type."""
        recommendations = []
        
        if not request.task_context:
            return recommendations
        
        task_type = request.task_context.task_type.lower()
        
        # Get capabilities for task type
        relevant_capabilities = self.task_type_rules.get(task_type, [])
        
        for cap_pattern in relevant_capabilities:
            # Search for capabilities matching the pattern
            search_query = SearchQuery(
                query=cap_pattern,
                search_mode=SearchMode.FUZZY,
                max_results=5
            )
            
            search_results = self.searcher.search(search_query)
            
            for result in search_results:
                justification = f"Recommended for {task_type} tasks based on capability pattern matching"
                
                recommendation = Recommendation(
                    capability=result.capability,
                    recommendation_type=RecommendationType.TASK_BASED,
                    confidence=self._determine_confidence(result.score),
                    score=result.score,
                    justification=justification,
                    pipeline_position=self._determine_pipeline_position(result.capability)
                )
                
                recommendations.append(recommendation)
        
        return recommendations
    
    def _get_data_driven_recommendations(self, request: RecommendationRequest) -> List[Recommendation]:
        """Get recommendations based on data characteristics."""
        recommendations = []
        
        if not request.data_characteristics:
            return recommendations
        
        data_chars = request.data_characteristics
        
        # Recommendations based on data types
        for data_type in data_chars.data_types:
            relevant_capabilities = self.data_type_rules.get(data_type.lower(), [])
            
            for cap_pattern in relevant_capabilities:
                search_query = SearchQuery(
                    query=cap_pattern,
                    search_mode=SearchMode.FUZZY,
                    max_results=3
                )
                
                search_results = self.searcher.search(search_query)
                
                for result in search_results:
                    justification = f"Recommended for {data_type} data processing"
                    
                    recommendation = Recommendation(
                        capability=result.capability,
                        recommendation_type=RecommendationType.DATA_DRIVEN,
                        confidence=self._determine_confidence(result.score),
                        score=result.score,
                        justification=justification
                    )
                    
                    recommendations.append(recommendation)
        
        # Recommendations based on data quality
        if data_chars.data_quality != "unknown":
            quality_capabilities = self.quality_rules.get(data_chars.data_quality, [])
            
            for cap_pattern in quality_capabilities:
                search_query = SearchQuery(
                    query=cap_pattern,
                    search_mode=SearchMode.FUZZY,
                    max_results=3
                )
                
                search_results = self.searcher.search(search_query)
                
                for result in search_results:
                    justification = f"Recommended for {data_chars.data_quality} quality data"
                    
                    recommendation = Recommendation(
                        capability=result.capability,
                        recommendation_type=RecommendationType.DATA_DRIVEN,
                        confidence=ConfidenceLevel.MEDIUM,
                        score=result.score * 0.8,  # Slightly lower score for quality-based
                        justification=justification
                    )
                    
                    recommendations.append(recommendation)
        
        # Special recommendations for specific data characteristics
        if data_chars.missing_values:
            search_query = SearchQuery(
                query="missing value imputation",
                search_mode=SearchMode.SEMANTIC,
                max_results=3
            )
            
            search_results = self.searcher.search(search_query)
            
            for result in search_results:
                recommendation = Recommendation(
                    capability=result.capability,
                    recommendation_type=RecommendationType.DATA_DRIVEN,
                    confidence=ConfidenceLevel.HIGH,
                    score=result.score * 1.2,  # Higher score for specific needs
                    justification="Recommended for handling missing values in your data"
                )
                
                recommendations.append(recommendation)
        
        return recommendations
    
    def _get_objective_oriented_recommendations(self, request: RecommendationRequest) -> List[Recommendation]:
        """Get recommendations based on objectives."""
        recommendations = []
        
        for objective in request.objectives:
            # Search for capabilities that help achieve this objective
            search_query = SearchQuery(
                query=objective,
                search_mode=SearchMode.SEMANTIC,
                max_results=5
            )
            
            search_results = self.searcher.search(search_query)
            
            for result in search_results:
                justification = f"Helps achieve objective: {objective}"
                
                recommendation = Recommendation(
                    capability=result.capability,
                    recommendation_type=RecommendationType.OBJECTIVE_ORIENTED,
                    confidence=self._determine_confidence(result.score),
                    score=result.score,
                    justification=justification
                )
                
                recommendations.append(recommendation)
        
        return recommendations
    
    def _get_pipeline_completion_recommendations(self, request: RecommendationRequest) -> List[Recommendation]:
        """Get recommendations to complete existing pipeline."""
        recommendations = []
        
        for current_cap_id in request.current_capabilities:
            if current_cap_id in self.index.capabilities:
                # Find what typically follows this capability
                compatible_caps = self.compatibility_matrix.get(current_cap_id, {}).get("precedes", [])
                
                for next_cap_pattern in compatible_caps:
                    search_query = SearchQuery(
                        query=next_cap_pattern,
                        search_mode=SearchMode.FUZZY,
                        max_results=3
                    )
                    
                    search_results = self.searcher.search(search_query)
                    
                    for result in search_results:
                        justification = f"Typically follows {current_cap_id} in processing pipelines"
                        
                        recommendation = Recommendation(
                            capability=result.capability,
                            recommendation_type=RecommendationType.PIPELINE_COMPLETION,
                            confidence=ConfidenceLevel.MEDIUM,
                            score=result.score * 0.9,
                            justification=justification
                        )
                        
                        recommendations.append(recommendation)
        
        return recommendations
    
    def _get_hybrid_recommendations(self, request: RecommendationRequest) -> List[Recommendation]:
        """Get recommendations using hybrid approach."""
        recommendations = []
        
        # Combine multiple strategies
        if request.task_context:
            recommendations.extend(self._get_task_based_recommendations(request))
        
        if request.data_characteristics:
            recommendations.extend(self._get_data_driven_recommendations(request))
        
        if request.objectives:
            recommendations.extend(self._get_objective_oriented_recommendations(request))
        
        if request.current_capabilities:
            recommendations.extend(self._get_pipeline_completion_recommendations(request))
        
        # Also do a general search on the query
        search_query = SearchQuery(
            query=request.query,
            search_mode=SearchMode.HYBRID,
            max_results=10
        )
        
        search_results = self.searcher.search(search_query)
        
        for result in search_results:
            justification = f"Matches your query: {request.query}"
            
            recommendation = Recommendation(
                capability=result.capability,
                recommendation_type=RecommendationType.TASK_BASED,
                confidence=self._determine_confidence(result.score),
                score=result.score,
                justification=justification
            )
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _get_alternative_recommendations(self, existing_recommendations: List[Recommendation], 
                                      request: RecommendationRequest) -> List[Recommendation]:
        """Get alternative capability recommendations."""
        alternatives = []
        
        for recommendation in existing_recommendations[:5]:  # Top 5 only
            # Find capabilities with similar functions
            related_results = self.searcher.get_related_capabilities(
                recommendation.capability.id, 
                limit=3
            )
            
            for related in related_results:
                if related.capability.id not in [r.capability.id for r in existing_recommendations]:
                    justification = f"Alternative to {recommendation.capability.name}"
                    
                    alternative = Recommendation(
                        capability=related.capability,
                        recommendation_type=RecommendationType.ALTERNATIVE,
                        confidence=ConfidenceLevel.LOW,
                        score=related.score * 0.7,  # Lower score for alternatives
                        justification=justification
                    )
                    
                    alternatives.append(alternative)
        
        return alternatives
    
    def _get_pipeline_suggestions(self, existing_recommendations: List[Recommendation],
                                request: RecommendationRequest) -> List[Recommendation]:
        """Get pipeline-based suggestions."""
        suggestions = []
        
        # Analyze the recommended capabilities and suggest pipeline improvements
        recommended_types = set()
        for rec in existing_recommendations:
            if rec.capability.categories:
                recommended_types.update(rec.capability.categories)
        
        # Check for common pipeline gaps
        common_pipeline_steps = [
            "data_validation",
            "feature_engineering", 
            "model_evaluation",
            "visualization",
            "documentation"
        ]
        
        for step in common_pipeline_steps:
            if step not in recommended_types:
                search_query = SearchQuery(
                    query=step,
                    search_mode=SearchMode.FUZZY,
                    max_results=2
                )
                
                search_results = self.searcher.search(search_query)
                
                for result in search_results:
                    justification = f"Suggested to complete your analysis pipeline"
                    
                    suggestion = Recommendation(
                        capability=result.capability,
                        recommendation_type=RecommendationType.PIPELINE_COMPLETION,
                        confidence=ConfidenceLevel.MEDIUM,
                        score=result.score * 0.6,
                        justification=justification
                    )
                    
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _determine_confidence(self, search_score: float) -> ConfidenceLevel:
        """Determine confidence level based on search score."""
        if search_score >= 0.8:
            return ConfidenceLevel.HIGH
        elif search_score >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _determine_pipeline_position(self, capability: CapabilityMetadata) -> Optional[str]:
        """Determine where capability fits in pipeline."""
        categories = [cat.lower() for cat in capability.categories]
        tags = [tag.lower() for tag in capability.tags]
        
        preprocessing_terms = ["cleaning", "preprocessing", "preparation", "transformation"]
        analysis_terms = ["analysis", "modeling", "training", "prediction"]
        postprocessing_terms = ["visualization", "reporting", "evaluation", "export"]
        
        all_terms = categories + tags
        
        if any(term in " ".join(all_terms) for term in preprocessing_terms):
            return "preprocessing"
        elif any(term in " ".join(all_terms) for term in analysis_terms):
            return "analysis"
        elif any(term in " ".join(all_terms) for term in postprocessing_terms):
            return "postprocessing"
        
        return None
    
    def _deduplicate_recommendations(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """Remove duplicate recommendations."""
        seen_ids = set()
        unique_recommendations = []
        
        for recommendation in recommendations:
            if recommendation.capability.id not in seen_ids:
                seen_ids.add(recommendation.capability.id)
                unique_recommendations.append(recommendation)
        
        return unique_recommendations
    
    def _score_recommendations(self, recommendations: List[Recommendation], 
                             request: RecommendationRequest) -> List[Recommendation]:
        """Apply additional scoring based on request preferences."""
        default_weights = {
            "relevance": 0.4,
            "popularity": 0.2,
            "success_rate": 0.2,
            "recency": 0.1,
            "complexity": 0.1
        }
        
        weights = {**default_weights, **request.preference_weights}
        
        for recommendation in recommendations:
            capability = recommendation.capability
            
            # Base score from search/recommendation
            base_score = recommendation.score
            
            # Popularity factor (usage count)
            popularity_score = min(capability.usage_count / 100.0, 1.0)  # Normalize to 0-1
            
            # Success rate factor
            success_score = capability.success_rate
            
            # Recency factor (how recently updated)
            days_old = (datetime.utcnow() - capability.last_updated).days
            recency_score = max(0, 1.0 - (days_old / 365.0))  # Decay over a year
            
            # Complexity factor (based on user experience)
            complexity_score = 1.0  # Default
            if request.task_context and request.task_context.user_experience == "beginner":
                # Prefer simpler capabilities for beginners
                if "simple" in capability.tags or "basic" in capability.tags:
                    complexity_score = 1.2
                elif "advanced" in capability.tags or "complex" in capability.tags:
                    complexity_score = 0.8
            
            # Calculate weighted final score
            final_score = (
                base_score * weights.get("relevance", 0.4) +
                popularity_score * weights.get("popularity", 0.2) +
                success_score * weights.get("success_rate", 0.2) +
                recency_score * weights.get("recency", 0.1) +
                complexity_score * weights.get("complexity", 0.1)
            )
            
            recommendation.score = final_score
        
        return recommendations
    
    def explain_recommendation(self, recommendation: Recommendation) -> Dict[str, Any]:
        """
        Provide detailed explanation for a recommendation.
        
        Args:
            recommendation: Recommendation to explain
            
        Returns:
            Detailed explanation
        """
        capability = recommendation.capability
        
        explanation = {
            "capability_name": capability.name,
            "recommendation_type": recommendation.recommendation_type.value,
            "confidence": recommendation.confidence.value,
            "score": recommendation.score,
            "primary_justification": recommendation.justification,
            "detailed_analysis": {
                "capability_type": capability.type.value,
                "categories": capability.categories,
                "tags": capability.tags,
                "usage_statistics": {
                    "usage_count": capability.usage_count,
                    "success_rate": capability.success_rate
                },
                "requirements": {
                    "inputs": len(capability.inputs),
                    "outputs": len(capability.outputs),
                    "dependencies": len(capability.dependencies)
                }
            },
            "pros": recommendation.pros or self._generate_pros(capability),
            "cons": recommendation.cons or self._generate_cons(capability),
            "alternatives": recommendation.alternatives,
            "usage_examples": recommendation.usage_examples or self._generate_usage_examples(capability),
            "implementation_details": {
                "estimated_effort": recommendation.estimated_effort,
                "expected_impact": recommendation.expected_impact,
                "pipeline_position": recommendation.pipeline_position
            }
        }
        
        return explanation
    
    def _generate_pros(self, capability: CapabilityMetadata) -> List[str]:
        """Generate pros for a capability."""
        pros = []
        
        if capability.success_rate > 0.8:
            pros.append("High success rate in previous usage")
        
        if capability.usage_count > 50:
            pros.append("Well-tested and frequently used")
        
        if "fast" in capability.tags or "optimized" in capability.tags:
            pros.append("Optimized for performance")
        
        if "easy" in capability.tags or "simple" in capability.tags:
            pros.append("Easy to use and configure")
        
        return pros or ["Matches your requirements"]
    
    def _generate_cons(self, capability: CapabilityMetadata) -> List[str]:
        """Generate cons for a capability."""
        cons = []
        
        if capability.success_rate < 0.5:
            cons.append("Lower success rate in previous usage")
        
        if capability.usage_count < 5:
            cons.append("Limited usage history")
        
        if "complex" in capability.tags or "advanced" in capability.tags:
            cons.append("May require advanced expertise")
        
        if len(capability.dependencies) > 5:
            cons.append("Has many dependencies")
        
        return cons
    
    def _generate_usage_examples(self, capability: CapabilityMetadata) -> List[str]:
        """Generate usage examples for a capability."""
        examples = []
        
        # Generate based on capability type and categories
        if capability.type == CapabilityType.METHOD:
            examples.append(f"Apply {capability.name} to your dataset")
            
        elif capability.type == CapabilityType.FUNCTION:
            examples.append(f"Use {capability.name} in your data pipeline")
            
        elif capability.type == CapabilityType.API:
            examples.append(f"Fetch data from {capability.name} API")
        
        return examples or [f"Use {capability.name} for your analysis"]
