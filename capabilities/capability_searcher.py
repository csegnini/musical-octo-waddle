"""
Capability Searcher Module

This module provides intelligent search capabilities across all indexed system capabilities
including fuzzy search, semantic search, and advanced filtering.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import difflib
from datetime import datetime

from .capability_indexer import CapabilityIndex, CapabilityMetadata, CapabilityType, CapabilityStatus

logger = logging.getLogger(__name__)


class SearchMode(Enum):
    """Search modes for capability discovery."""
    EXACT = "exact"
    FUZZY = "fuzzy"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class SearchScope(Enum):
    """Scope for capability search."""
    ALL = "all"
    NAME = "name"
    DESCRIPTION = "description"
    TAGS = "tags"
    CATEGORIES = "categories"


class SortBy(Enum):
    """Sorting options for search results."""
    RELEVANCE = "relevance"
    NAME = "name"
    TYPE = "type"
    USAGE_COUNT = "usage_count"
    SUCCESS_RATE = "success_rate"
    LAST_UPDATED = "last_updated"


@dataclass
class SearchQuery:
    """Search query configuration."""
    query: str
    capability_types: Optional[List[CapabilityType]] = None
    tags: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    agents: Optional[List[str]] = None
    status_filter: Optional[List[CapabilityStatus]] = None
    search_mode: SearchMode = SearchMode.HYBRID
    search_scope: SearchScope = SearchScope.ALL
    sort_by: SortBy = SortBy.RELEVANCE
    max_results: int = 50
    min_score: float = 0.0
    include_dependencies: bool = False
    fuzzy_threshold: float = 0.6
    
    def __post_init__(self):
        """Validate query parameters."""
        if not self.query.strip():
            raise ValueError("Query cannot be empty")
        
        self.query = self.query.strip().lower()


@dataclass
class SearchResult:
    """Search result for a capability."""
    capability: CapabilityMetadata
    score: float
    match_details: Dict[str, Any] = field(default_factory=dict)
    highlighted_fields: Dict[str, str] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"{self.capability.name} ({self.capability.type.value}) - Score: {self.score:.3f}"


class CapabilitySearcher:
    """
    Intelligent search engine for system capabilities.
    
    Provides multiple search modes including exact matching, fuzzy search,
    and semantic search with advanced filtering and ranking.
    """
    
    def __init__(self, capability_index: CapabilityIndex):
        """
        Initialize capability searcher.
        
        Args:
            capability_index: Indexed capability data
        """
        self.index = capability_index
        self._build_search_indexes()
        
        logger.info(f"Initialized capability searcher with {len(self.index.capabilities)} capabilities")
    
    def _build_search_indexes(self) -> None:
        """Build optimized search indexes."""
        logger.debug("Building search indexes...")
        
        # Build text indexes for faster searching
        self.name_index = {}
        self.description_index = {}
        self.combined_text_index = {}
        
        for cap_id, capability in self.index.capabilities.items():
            # Name index
            name_tokens = self._tokenize(capability.name)
            for token in name_tokens:
                if token not in self.name_index:
                    self.name_index[token] = set()
                self.name_index[token].add(cap_id)
            
            # Description index
            desc_tokens = self._tokenize(capability.description)
            for token in desc_tokens:
                if token not in self.description_index:
                    self.description_index[token] = set()
                self.description_index[token].add(cap_id)
            
            # Combined text index
            all_text = f"{capability.name} {capability.description} {' '.join(capability.tags)}"
            combined_tokens = self._tokenize(all_text)
            for token in combined_tokens:
                if token not in self.combined_text_index:
                    self.combined_text_index[token] = set()
                self.combined_text_index[token].add(cap_id)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for indexing."""
        if not text:
            return []
        
        # Convert to lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
        
        return tokens
    
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """
        Search for capabilities based on query.
        
        Args:
            query: Search query configuration
            
        Returns:
            List of search results sorted by relevance
        """
        logger.debug(f"Searching for: '{query.query}' with mode: {query.search_mode.value}")
        
        # Apply filters first to narrow down candidates
        candidates = self._apply_filters(query)
        
        if not candidates:
            logger.debug("No candidates after filtering")
            return []
        
        # Score candidates based on search mode
        scored_results = []
        
        if query.search_mode == SearchMode.EXACT:
            scored_results = self._exact_search(query, candidates)
        elif query.search_mode == SearchMode.FUZZY:
            scored_results = self._fuzzy_search(query, candidates)
        elif query.search_mode == SearchMode.SEMANTIC:
            scored_results = self._semantic_search(query, candidates)
        elif query.search_mode == SearchMode.HYBRID:
            scored_results = self._hybrid_search(query, candidates)
        
        # Filter by minimum score
        scored_results = [result for result in scored_results if result.score >= query.min_score]
        
        # Sort results
        scored_results = self._sort_results(scored_results, query.sort_by)
        
        # Limit results
        if query.max_results > 0:
            scored_results = scored_results[:query.max_results]
        
        # Add dependencies if requested
        if query.include_dependencies:
            scored_results = self._include_dependencies(scored_results)
        
        logger.debug(f"Found {len(scored_results)} results")
        return scored_results
    
    def _apply_filters(self, query: SearchQuery) -> Set[str]:
        """Apply filters to get candidate capability IDs."""
        candidates = set(self.index.capabilities.keys())
        
        # Filter by capability types
        if query.capability_types:
            type_candidates = set()
            for cap_type in query.capability_types:
                type_candidates.update(self.index.type_index.get(cap_type, set()))
            candidates &= type_candidates
        
        # Filter by tags
        if query.tags:
            tag_candidates = set()
            for tag in query.tags:
                tag_lower = tag.lower()
                tag_candidates.update(self.index.tag_index.get(tag_lower, set()))
            candidates &= tag_candidates
        
        # Filter by categories
        if query.categories:
            category_candidates = set()
            for category in query.categories:
                category_candidates.update(self.index.category_index.get(category, set()))
            candidates &= category_candidates
        
        # Filter by responsible agents
        if query.agents:
            agent_candidates = set()
            for agent in query.agents:
                agent_candidates.update(self.index.agent_capabilities.get(agent, set()))
            candidates &= agent_candidates
        
        # Filter by status
        if query.status_filter:
            status_candidates = set()
            for cap_id in candidates:
                capability = self.index.capabilities[cap_id]
                if capability.status in query.status_filter:
                    status_candidates.add(cap_id)
            candidates = status_candidates
        
        return candidates
    
    def _exact_search(self, query: SearchQuery, candidates: Set[str]) -> List[SearchResult]:
        """Perform exact string matching search."""
        results = []
        query_lower = query.query.lower()
        
        for cap_id in candidates:
            capability = self.index.capabilities[cap_id]
            score = 0.0
            match_details = {}
            highlighted = {}
            
            # Check exact matches in different fields
            if query.search_scope in [SearchScope.ALL, SearchScope.NAME]:
                if query_lower in capability.name.lower():
                    score += 1.0
                    match_details["name_match"] = True
                    highlighted["name"] = self._highlight_match(capability.name, query.query)
            
            if query.search_scope in [SearchScope.ALL, SearchScope.DESCRIPTION]:
                if query_lower in capability.description.lower():
                    score += 0.8
                    match_details["description_match"] = True
                    highlighted["description"] = self._highlight_match(capability.description, query.query)
            
            if query.search_scope in [SearchScope.ALL, SearchScope.TAGS]:
                for tag in capability.tags:
                    if query_lower in tag.lower():
                        score += 0.6
                        match_details["tag_match"] = True
                        break
            
            if query.search_scope in [SearchScope.ALL, SearchScope.CATEGORIES]:
                for category in capability.categories:
                    if query_lower in category.lower():
                        score += 0.5
                        match_details["category_match"] = True
                        break
            
            if score > 0:
                results.append(SearchResult(
                    capability=capability,
                    score=score,
                    match_details=match_details,
                    highlighted_fields=highlighted
                ))
        
        return results
    
    def _fuzzy_search(self, query: SearchQuery, candidates: Set[str]) -> List[SearchResult]:
        """Perform fuzzy string matching search."""
        results = []
        query_tokens = self._tokenize(query.query)
        
        for cap_id in candidates:
            capability = self.index.capabilities[cap_id]
            total_score = 0.0
            match_details = {}
            highlighted = {}
            
            # Fuzzy match against name
            if query.search_scope in [SearchScope.ALL, SearchScope.NAME]:
                name_score = self._fuzzy_match_text(query.query, capability.name, query.fuzzy_threshold)
                if name_score > 0:
                    total_score += name_score * 1.0
                    match_details["name_fuzzy_score"] = name_score
                    highlighted["name"] = capability.name
            
            # Fuzzy match against description
            if query.search_scope in [SearchScope.ALL, SearchScope.DESCRIPTION]:
                desc_score = self._fuzzy_match_text(query.query, capability.description, query.fuzzy_threshold)
                if desc_score > 0:
                    total_score += desc_score * 0.8
                    match_details["description_fuzzy_score"] = desc_score
                    highlighted["description"] = capability.description[:200] + "..."
            
            # Token-based fuzzy matching
            token_matches = 0
            for query_token in query_tokens:
                # Check tags
                for tag in capability.tags:
                    if difflib.SequenceMatcher(None, query_token, tag.lower()).ratio() >= query.fuzzy_threshold:
                        token_matches += 1
                        break
                
                # Check categories
                for category in capability.categories:
                    if difflib.SequenceMatcher(None, query_token, category.lower()).ratio() >= query.fuzzy_threshold:
                        token_matches += 1
                        break
            
            if token_matches > 0:
                total_score += (token_matches / len(query_tokens)) * 0.5
                match_details["token_matches"] = token_matches
            
            if total_score > 0:
                results.append(SearchResult(
                    capability=capability,
                    score=total_score,
                    match_details=match_details,
                    highlighted_fields=highlighted
                ))
        
        return results
    
    def _fuzzy_match_text(self, query: str, text: str, threshold: float) -> float:
        """Calculate fuzzy match score between query and text."""
        if not query or not text:
            return 0.0
        
        # Direct similarity
        similarity = difflib.SequenceMatcher(None, query.lower(), text.lower()).ratio()
        
        if similarity >= threshold:
            return similarity
        
        # Token-based similarity
        query_tokens = self._tokenize(query)
        text_tokens = self._tokenize(text)
        
        if not query_tokens or not text_tokens:
            return similarity if similarity >= threshold else 0.0
        
        max_token_score = 0.0
        for q_token in query_tokens:
            for t_token in text_tokens:
                token_sim = difflib.SequenceMatcher(None, q_token, t_token).ratio()
                max_token_score = max(max_token_score, token_sim)
        
        combined_score = (similarity + max_token_score) / 2
        return combined_score if combined_score >= threshold else 0.0
    
    def _semantic_search(self, query: SearchQuery, candidates: Set[str]) -> List[SearchResult]:
        """Perform semantic search using word embeddings and context."""
        # For now, implement a simple semantic search using keyword expansion
        # In a full implementation, this would use word embeddings or transformers
        
        results = []
        
        # Expand query with synonyms and related terms
        expanded_query = self._expand_query_semantically(query.query)
        
        for cap_id in candidates:
            capability = self.index.capabilities[cap_id]
            score = 0.0
            match_details = {}
            
            # Score based on semantic similarity
            all_text = f"{capability.name} {capability.description} {' '.join(capability.tags)} {' '.join(capability.categories)}"
            
            for expanded_term in expanded_query:
                if expanded_term.lower() in all_text.lower():
                    score += 0.7  # Lower score for expanded terms
                    match_details[f"semantic_match_{expanded_term}"] = True
            
            # Original query terms get higher scores
            original_tokens = self._tokenize(query.query)
            for token in original_tokens:
                if token in all_text.lower():
                    score += 1.0
                    match_details[f"original_match_{token}"] = True
            
            if score > 0:
                # Normalize score
                score = score / (len(expanded_query) + len(original_tokens))
                
                results.append(SearchResult(
                    capability=capability,
                    score=score,
                    match_details=match_details,
                    highlighted_fields={}
                ))
        
        return results
    
    def _expand_query_semantically(self, query: str) -> List[str]:
        """Expand query with semantically related terms."""
        # Simple keyword expansion - in production, use word embeddings
        expansion_map = {
            "data": ["dataset", "information", "records", "table"],
            "clean": ["preprocessing", "cleanse", "prepare", "transform"],
            "analyze": ["analysis", "examine", "study", "investigate"],
            "model": ["algorithm", "prediction", "machine learning", "ml"],
            "visualize": ["plot", "chart", "graph", "display"],
            "load": ["import", "read", "fetch", "get"],
            "save": ["export", "write", "store", "persist"],
            "train": ["fit", "learn", "optimize", "tune"],
            "predict": ["forecast", "estimate", "infer", "classify"]
        }
        
        expanded_terms = []
        query_tokens = self._tokenize(query)
        
        for token in query_tokens:
            if token in expansion_map:
                expanded_terms.extend(expansion_map[token])
        
        return expanded_terms
    
    def _hybrid_search(self, query: SearchQuery, candidates: Set[str]) -> List[SearchResult]:
        """Perform hybrid search combining multiple search modes."""
        # Get results from different search modes
        exact_results = self._exact_search(query, candidates)
        fuzzy_results = self._fuzzy_search(query, candidates)
        semantic_results = self._semantic_search(query, candidates)
        
        # Combine results with weighted scores
        combined_results = {}
        
        # Exact matches get highest weight
        for result in exact_results:
            cap_id = result.capability.id
            combined_results[cap_id] = SearchResult(
                capability=result.capability,
                score=result.score * 1.0,  # Full weight
                match_details={"exact": result.match_details},
                highlighted_fields=result.highlighted_fields
            )
        
        # Add fuzzy results with medium weight
        for result in fuzzy_results:
            cap_id = result.capability.id
            if cap_id in combined_results:
                # Merge scores
                combined_results[cap_id].score += result.score * 0.7
                combined_results[cap_id].match_details["fuzzy"] = result.match_details
            else:
                combined_results[cap_id] = SearchResult(
                    capability=result.capability,
                    score=result.score * 0.7,
                    match_details={"fuzzy": result.match_details},
                    highlighted_fields=result.highlighted_fields
                )
        
        # Add semantic results with lower weight
        for result in semantic_results:
            cap_id = result.capability.id
            if cap_id in combined_results:
                # Merge scores
                combined_results[cap_id].score += result.score * 0.5
                combined_results[cap_id].match_details["semantic"] = result.match_details
            else:
                combined_results[cap_id] = SearchResult(
                    capability=result.capability,
                    score=result.score * 0.5,
                    match_details={"semantic": result.match_details},
                    highlighted_fields={}
                )
        
        return list(combined_results.values())
    
    def _highlight_match(self, text: str, query: str) -> str:
        """Highlight matching terms in text."""
        if not query or not text:
            return text
        
        # Simple highlighting - in production, use more sophisticated highlighting
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        return pattern.sub(f"**{query}**", text)
    
    def _sort_results(self, results: List[SearchResult], sort_by: SortBy) -> List[SearchResult]:
        """Sort search results based on criteria."""
        if sort_by == SortBy.RELEVANCE:
            return sorted(results, key=lambda r: r.score, reverse=True)
        elif sort_by == SortBy.NAME:
            return sorted(results, key=lambda r: r.capability.name.lower())
        elif sort_by == SortBy.TYPE:
            return sorted(results, key=lambda r: r.capability.type.value)
        elif sort_by == SortBy.USAGE_COUNT:
            return sorted(results, key=lambda r: r.capability.usage_count, reverse=True)
        elif sort_by == SortBy.SUCCESS_RATE:
            return sorted(results, key=lambda r: r.capability.success_rate, reverse=True)
        elif sort_by == SortBy.LAST_UPDATED:
            return sorted(results, key=lambda r: r.capability.last_updated, reverse=True)
        
        return results
    
    def _include_dependencies(self, results: List[SearchResult]) -> List[SearchResult]:
        """Include capability dependencies in results."""
        extended_results = list(results)
        
        for result in results:
            cap_id = result.capability.id
            
            # Add dependencies
            dependencies = self.index.dependency_graph.get(cap_id, set())
            for dep_id in dependencies:
                if dep_id in self.index.capabilities:
                    dep_capability = self.index.capabilities[dep_id]
                    # Add dependency with lower score
                    dep_result = SearchResult(
                        capability=dep_capability,
                        score=result.score * 0.3,
                        match_details={"dependency_of": cap_id},
                        highlighted_fields={}
                    )
                    extended_results.append(dep_result)
        
        return extended_results
    
    def suggest_capabilities(self, partial_query: str, limit: int = 10) -> List[str]:
        """
        Suggest capability names based on partial query.
        
        Args:
            partial_query: Partial search query
            limit: Maximum number of suggestions
            
        Returns:
            List of suggested capability names
        """
        if not partial_query:
            return []
        
        partial_lower = partial_query.lower()
        suggestions = []
        
        for capability in self.index.capabilities.values():
            if partial_lower in capability.name.lower():
                suggestions.append(capability.name)
        
        # Sort by length (shorter names first) and alphabetically
        suggestions.sort(key=lambda x: (len(x), x.lower()))
        
        return suggestions[:limit]
    
    def get_related_capabilities(self, capability_id: str, limit: int = 10) -> List[SearchResult]:
        """
        Get capabilities related to the given capability.
        
        Args:
            capability_id: ID of the capability to find relations for
            limit: Maximum number of related capabilities
            
        Returns:
            List of related capabilities
        """
        if capability_id not in self.index.capabilities:
            return []
        
        capability = self.index.capabilities[capability_id]
        related = []
        
        # Find capabilities with similar tags
        for other_id, other_cap in self.index.capabilities.items():
            if other_id == capability_id:
                continue
            
            # Calculate similarity based on shared tags
            shared_tags = set(capability.tags) & set(other_cap.tags)
            tag_similarity = len(shared_tags) / max(len(capability.tags), len(other_cap.tags), 1)
            
            # Calculate similarity based on shared categories
            shared_categories = set(capability.categories) & set(other_cap.categories)
            category_similarity = len(shared_categories) / max(len(capability.categories), len(other_cap.categories), 1)
            
            # Calculate similarity based on same type
            type_similarity = 1.0 if capability.type == other_cap.type else 0.0
            
            # Combined similarity score
            similarity = (tag_similarity * 0.5 + category_similarity * 0.3 + type_similarity * 0.2)
            
            if similarity > 0.1:  # Minimum threshold
                related.append(SearchResult(
                    capability=other_cap,
                    score=similarity,
                    match_details={
                        "shared_tags": list(shared_tags),
                        "shared_categories": list(shared_categories),
                        "tag_similarity": tag_similarity,
                        "category_similarity": category_similarity
                    }
                ))
        
        # Sort by similarity score
        related.sort(key=lambda r: r.score, reverse=True)
        
        return related[:limit]
