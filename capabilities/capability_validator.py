"""
Capability Validator Module

This module provides validation capabilities for ensuring capability
definitions are correct and complete.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from .capability_indexer import CapabilityMetadata, CapabilityType, CapabilityStatus

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Validation issue found in capability definition."""
    capability_id: str
    field: str
    severity: ValidationSeverity
    message: str
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of capability validation."""
    capability_id: str
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    score: float = 0.0  # 0-100 quality score
    
    def has_errors(self) -> bool:
        """Check if validation result has errors."""
        return any(issue.severity == ValidationSeverity.ERROR for issue in self.issues)
    
    def has_warnings(self) -> bool:
        """Check if validation result has warnings."""
        return any(issue.severity == ValidationSeverity.WARNING for issue in self.issues)


class CapabilityValidator:
    """
    Validator for capability definitions.
    
    This class validates capability definitions for completeness,
    correctness, and best practices.
    """
    
    def __init__(self):
        """Initialize capability validator."""
        self.validation_rules = self._build_validation_rules()
        logger.info("Initialized capability validator")
    
    def _build_validation_rules(self) -> Dict[str, Any]:
        """Build validation rules for capabilities."""
        return {
            "required_fields": ["id", "name", "description", "type"],
            "id_pattern": r"^[a-z][a-z0-9_]*$",
            "name_min_length": 3,
            "name_max_length": 100,
            "description_min_length": 10,
            "description_max_length": 500,
            "max_tags": 10,
            "max_categories": 5,
            "max_dependencies": 20
        }
    
    def validate_capability(self, capability: CapabilityMetadata) -> ValidationResult:
        """
        Validate a single capability.
        
        Args:
            capability: Capability to validate
            
        Returns:
            Validation result
        """
        result = ValidationResult(capability_id=capability.id, is_valid=True)
        
        # Validate required fields
        self._validate_required_fields(capability, result)
        
        # Validate ID format
        self._validate_id_format(capability, result)
        
        # Validate name
        self._validate_name(capability, result)
        
        # Validate description
        self._validate_description(capability, result)
        
        # Validate tags and categories
        self._validate_tags_and_categories(capability, result)
        
        # Validate dependencies
        self._validate_dependencies(capability, result)
        
        # Validate type-specific requirements
        self._validate_type_specific(capability, result)
        
        # Calculate quality score
        result.score = self._calculate_quality_score(capability, result)
        
        # Determine overall validity
        result.is_valid = not result.has_errors()
        
        return result
    
    def _validate_required_fields(self, capability: CapabilityMetadata, result: ValidationResult) -> None:
        """Validate required fields are present."""
        required_fields = self.validation_rules["required_fields"]
        
        for field in required_fields:
            value = getattr(capability, field, None)
            if not value or (isinstance(value, str) and not value.strip()):
                result.issues.append(ValidationIssue(
                    capability_id=capability.id,
                    field=field,
                    severity=ValidationSeverity.ERROR,
                    message=f"Required field '{field}' is missing or empty",
                    suggestion=f"Provide a valid value for {field}"
                ))
    
    def _validate_id_format(self, capability: CapabilityMetadata, result: ValidationResult) -> None:
        """Validate ID format."""
        import re
        
        pattern = self.validation_rules["id_pattern"]
        if not re.match(pattern, capability.id):
            result.issues.append(ValidationIssue(
                capability_id=capability.id,
                field="id",
                severity=ValidationSeverity.ERROR,
                message="ID must start with lowercase letter and contain only lowercase letters, numbers, and underscores",
                suggestion="Use format like 'data_cleaning' or 'model_training'"
            ))
    
    def _validate_name(self, capability: CapabilityMetadata, result: ValidationResult) -> None:
        """Validate name field."""
        name = capability.name
        min_length = self.validation_rules["name_min_length"]
        max_length = self.validation_rules["name_max_length"]
        
        if len(name) < min_length:
            result.issues.append(ValidationIssue(
                capability_id=capability.id,
                field="name",
                severity=ValidationSeverity.WARNING,
                message=f"Name is too short (minimum {min_length} characters)",
                suggestion="Provide a more descriptive name"
            ))
        
        if len(name) > max_length:
            result.issues.append(ValidationIssue(
                capability_id=capability.id,
                field="name",
                severity=ValidationSeverity.WARNING,
                message=f"Name is too long (maximum {max_length} characters)",
                suggestion="Shorten the name while keeping it descriptive"
            ))
    
    def _validate_description(self, capability: CapabilityMetadata, result: ValidationResult) -> None:
        """Validate description field."""
        description = capability.description
        min_length = self.validation_rules["description_min_length"]
        max_length = self.validation_rules["description_max_length"]
        
        if len(description) < min_length:
            result.issues.append(ValidationIssue(
                capability_id=capability.id,
                field="description",
                severity=ValidationSeverity.WARNING,
                message=f"Description is too short (minimum {min_length} characters)",
                suggestion="Provide a more detailed description of what this capability does"
            ))
        
        if len(description) > max_length:
            result.issues.append(ValidationIssue(
                capability_id=capability.id,
                field="description",
                severity=ValidationSeverity.INFO,
                message=f"Description is very long ({len(description)} characters)",
                suggestion="Consider shortening while keeping essential information"
            ))
    
    def _validate_tags_and_categories(self, capability: CapabilityMetadata, result: ValidationResult) -> None:
        """Validate tags and categories."""
        max_tags = self.validation_rules["max_tags"]
        max_categories = self.validation_rules["max_categories"]
        
        if len(capability.tags) > max_tags:
            result.issues.append(ValidationIssue(
                capability_id=capability.id,
                field="tags",
                severity=ValidationSeverity.WARNING,
                message=f"Too many tags ({len(capability.tags)}, maximum {max_tags})",
                suggestion="Keep only the most relevant tags"
            ))
        
        if len(capability.categories) > max_categories:
            result.issues.append(ValidationIssue(
                capability_id=capability.id,
                field="categories",
                severity=ValidationSeverity.WARNING,
                message=f"Too many categories ({len(capability.categories)}, maximum {max_categories})",
                suggestion="Organize into fewer, more general categories"
            ))
        
        if not capability.tags:
            result.issues.append(ValidationIssue(
                capability_id=capability.id,
                field="tags",
                severity=ValidationSeverity.INFO,
                message="No tags specified",
                suggestion="Add relevant tags to improve discoverability"
            ))
    
    def _validate_dependencies(self, capability: CapabilityMetadata, result: ValidationResult) -> None:
        """Validate dependencies."""
        max_deps = self.validation_rules["max_dependencies"]
        
        if len(capability.dependencies) > max_deps:
            result.issues.append(ValidationIssue(
                capability_id=capability.id,
                field="dependencies",
                severity=ValidationSeverity.WARNING,
                message=f"Too many dependencies ({len(capability.dependencies)}, maximum {max_deps})",
                suggestion="Consider breaking into smaller capabilities with fewer dependencies"
            ))
    
    def _validate_type_specific(self, capability: CapabilityMetadata, result: ValidationResult) -> None:
        """Validate type-specific requirements."""
        if capability.type == CapabilityType.METHOD:
            if not capability.inputs:
                result.issues.append(ValidationIssue(
                    capability_id=capability.id,
                    field="inputs",
                    severity=ValidationSeverity.WARNING,
                    message="Method has no defined inputs",
                    suggestion="Define input parameters for the method"
                ))
            
            if not capability.outputs:
                result.issues.append(ValidationIssue(
                    capability_id=capability.id,
                    field="outputs",
                    severity=ValidationSeverity.WARNING,
                    message="Method has no defined outputs",
                    suggestion="Define expected outputs for the method"
                ))
        
        elif capability.type == CapabilityType.AGENT:
            if not capability.responsible_agents:
                result.issues.append(ValidationIssue(
                    capability_id=capability.id,
                    field="responsible_agents",
                    severity=ValidationSeverity.INFO,
                    message="Agent has no defined capabilities",
                    suggestion="Define what capabilities this agent provides"
                ))
        
        elif capability.type == CapabilityType.API:
            if not capability.parameters:
                result.issues.append(ValidationIssue(
                    capability_id=capability.id,
                    field="parameters",
                    severity=ValidationSeverity.WARNING,
                    message="API has no defined parameters",
                    suggestion="Define API configuration parameters"
                ))
    
    def _calculate_quality_score(self, capability: CapabilityMetadata, result: ValidationResult) -> float:
        """Calculate quality score for capability."""
        score = 100.0
        
        # Deduct points for issues
        for issue in result.issues:
            if issue.severity == ValidationSeverity.ERROR:
                score -= 20
            elif issue.severity == ValidationSeverity.WARNING:
                score -= 10
            elif issue.severity == ValidationSeverity.INFO:
                score -= 5
        
        # Bonus points for completeness
        if capability.inputs:
            score += 5
        if capability.outputs:
            score += 5
        if capability.tags:
            score += 5
        if capability.categories:
            score += 5
        if capability.version != "1.0.0":  # Non-default version
            score += 5
        
        return max(0.0, min(100.0, score))
    
    def validate_all_capabilities(self, capabilities: Dict[str, CapabilityMetadata]) -> Dict[str, ValidationResult]:
        """
        Validate all capabilities.
        
        Args:
            capabilities: Dictionary of capabilities to validate
            
        Returns:
            Dictionary of validation results
        """
        results = {}
        
        for cap_id, capability in capabilities.items():
            results[cap_id] = self.validate_capability(capability)
        
        logger.info(f"Validated {len(results)} capabilities")
        return results
    
    def get_validation_summary(self, results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """
        Get summary of validation results.
        
        Args:
            results: Validation results
            
        Returns:
            Summary statistics
        """
        total = len(results)
        valid = sum(1 for r in results.values() if r.is_valid)
        invalid = total - valid
        
        errors = sum(1 for r in results.values() if r.has_errors())
        warnings = sum(1 for r in results.values() if r.has_warnings())
        
        avg_score = sum(r.score for r in results.values()) / total if total > 0 else 0
        
        return {
            "total_capabilities": total,
            "valid_capabilities": valid,
            "invalid_capabilities": invalid,
            "capabilities_with_errors": errors,
            "capabilities_with_warnings": warnings,
            "average_quality_score": round(avg_score, 2),
            "validation_rate": round((valid / total) * 100, 2) if total > 0 else 0
        }
