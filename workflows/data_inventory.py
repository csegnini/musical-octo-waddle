"""
Data Inventory Management Module

This module provides comprehensive data inventory tracking, data reference management,
and data provenance tracking for workflow orchestration systems.
"""

import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import os

logger = logging.getLogger(__name__)


class DataType(Enum):
    """Enumeration of data types in the inventory."""
    UPLOADED = "uploaded_data_references"
    EXTRACTED = "extracted_data_references"
    PREPARED = "prepared_data_references"
    TRANSFORMED = "transformed_data_references"
    INTERMEDIATE = "intermediate_data_references"
    FINAL = "final_data_references"


class DataFormat(Enum):
    """Enumeration of supported data formats."""
    CSV = "text/csv"
    JSON = "application/json"
    PARQUET = "application/parquet"
    EXCEL = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    PICKLE = "application/octet-stream"
    FEATHER = "application/octet-stream"
    HDF5 = "application/x-hdf"
    SQL = "application/sql"


@dataclass
class ProvenanceInfo:
    """Represents data provenance information."""
    original_source: str = ""
    extracted_from: str = ""
    source_stage: str = ""
    parent_data_ids: List[str] = field(default_factory=list)
    transformation_history: List[str] = field(default_factory=list)
    creation_method: str = ""
    
    def add_transformation(self, transformation: str) -> None:
        """Add a transformation to the history."""
        self.transformation_history.append(transformation)
        logger.debug(f"Added transformation: {transformation}")


@dataclass
class DataReference:
    """
    Comprehensive data reference with metadata and provenance tracking.
    """
    name: str
    path: str
    data_id: Optional[str] = None
    size_bytes: int = 0
    record_count: int = 0
    mime_type: str = ""
    checksum_sha256: str = ""
    generated_by_step_id: str = ""
    timestamp_utc: str = ""
    description: str = ""
    schema_path: str = ""
    provenance_info: ProvenanceInfo = field(default_factory=ProvenanceInfo)
    transformations_applied: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize computed fields after creation."""
        if not self.data_id:
            self.data_id = self._generate_data_id()
        if not self.timestamp_utc:
            self.timestamp_utc = datetime.utcnow().isoformat() + "Z"
        if not self.checksum_sha256 and os.path.exists(self.path):
            self.checksum_sha256 = self._calculate_checksum()
        if self.size_bytes == 0 and os.path.exists(self.path):
            self.size_bytes = os.path.getsize(self.path)
    
    def _generate_data_id(self) -> str:
        """Generate a unique data ID based on path and timestamp."""
        content = f"{self.path}_{self.timestamp_utc}_{self.name}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum of the file."""
        try:
            hash_sha256 = hashlib.sha256()
            with open(self.path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.warning(f"Could not calculate checksum for {self.path}: {e}")
            return ""
    
    def update_metadata(self, key: str, value: Any) -> None:
        """Update metadata for the data reference."""
        self.metadata[key] = value
        logger.debug(f"Updated metadata {key} for data {self.data_id}")
    
    def add_transformation(self, transformation: str) -> None:
        """Add a transformation to the applied transformations list."""
        self.transformations_applied.append(transformation)
        self.provenance_info.add_transformation(transformation)
        logger.info(f"Added transformation '{transformation}' to data {self.data_id}")
    
    def validate_file_exists(self) -> bool:
        """Validate that the referenced file exists."""
        exists = os.path.exists(self.path)
        if not exists:
            logger.warning(f"Data file not found: {self.path}")
        return exists
    
    def get_file_stats(self) -> Dict[str, Any]:
        """Get file statistics if file exists."""
        if not self.validate_file_exists():
            return {}
        
        try:
            stat = os.stat(self.path)
            return {
                'size_bytes': stat.st_size,
                'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'accessed_time': datetime.fromtimestamp(stat.st_atime).isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting file stats for {self.path}: {e}")
            return {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert data reference to dictionary."""
        return {
            'data_id': self.data_id,
            'name': self.name,
            'path': self.path,
            'size_bytes': self.size_bytes,
            'record_count': self.record_count,
            'mime_type': self.mime_type,
            'checksum_sha256': self.checksum_sha256,
            'generated_by_step_id': self.generated_by_step_id,
            'timestamp_utc': self.timestamp_utc,
            'description': self.description,
            'schema_path': self.schema_path,
            'provenance_info': self.provenance_info.__dict__,
            'transformations_applied': self.transformations_applied,
            'metadata': self.metadata,
            'file_stats': self.get_file_stats()
        }


class DataInventoryManager:
    """
    Comprehensive data inventory manager for workflow orchestration.
    
    This class manages all data references throughout the workflow lifecycle,
    tracks data lineage, handles data validation, and provides data discovery capabilities.
    """
    
    def __init__(self, workflow_id: str):
        """
        Initialize data inventory manager.
        
        Args:
            workflow_id: Unique identifier for the workflow
        """
        self.workflow_id = workflow_id
        self.data_inventory: Dict[DataType, Dict[str, DataReference]] = {
            data_type: {} for data_type in DataType
        }
        self.creation_timestamp = datetime.utcnow().isoformat() + "Z"
        self.last_update_timestamp = self.creation_timestamp
        
        logger.info(f"Initialized data inventory manager for workflow {workflow_id}")
    
    def add_data_reference(
        self, 
        data_type: DataType, 
        data_ref: DataReference
    ) -> str:
        """
        Add a data reference to the inventory.
        
        Args:
            data_type: Type of data (uploaded, extracted, etc.)
            data_ref: Data reference object
            
        Returns:
            Data ID of the added reference
        """
        data_id = data_ref.data_id or data_ref._generate_data_id()
        data_ref.data_id = data_id
        self.data_inventory[data_type][data_id] = data_ref
        self.last_update_timestamp = datetime.utcnow().isoformat() + "Z"
        
        logger.info(f"Added {data_type.value} data reference: {data_ref.name} ({data_id})")
        return data_id
    
    def create_and_add_data_reference(
        self,
        data_type: DataType,
        name: str,
        path: str,
        description: str = "",
        generated_by_step_id: str = "",
        **kwargs
    ) -> DataReference:
        """
        Create and add a new data reference.
        
        Args:
            data_type: Type of data
            name: Name of the data file
            path: File path
            description: Description of the data
            generated_by_step_id: ID of the step that generated this data
            **kwargs: Additional parameters for DataReference
            
        Returns:
            Created DataReference object
        """
        data_ref = DataReference(
            name=name,
            path=path,
            description=description,
            generated_by_step_id=generated_by_step_id,
            **kwargs
        )
        
        self.add_data_reference(data_type, data_ref)
        return data_ref
    
    def get_data_reference(self, data_type: DataType, data_id: str) -> Optional[DataReference]:
        """
        Get a specific data reference.
        
        Args:
            data_type: Type of data
            data_id: Data reference ID
            
        Returns:
            DataReference object or None if not found
        """
        return self.data_inventory[data_type].get(data_id)
    
    def get_data_references_by_type(self, data_type: DataType) -> Dict[str, DataReference]:
        """Get all data references of a specific type."""
        return self.data_inventory[data_type].copy()
    
    def get_data_references_by_step(self, step_id: str) -> List[DataReference]:
        """Get all data references generated by a specific step."""
        references = []
        for data_type in DataType:
            for data_ref in self.data_inventory[data_type].values():
                if data_ref.generated_by_step_id == step_id:
                    references.append(data_ref)
        return references
    
    def search_data_references(
        self,
        name_pattern: Optional[str] = None,
        description_pattern: Optional[str] = None,
        mime_type: Optional[str] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None
    ) -> List[DataReference]:
        """
        Search data references based on various criteria.
        
        Args:
            name_pattern: Pattern to match in data name
            description_pattern: Pattern to match in description
            mime_type: MIME type to filter by
            min_size: Minimum file size in bytes
            max_size: Maximum file size in bytes
            
        Returns:
            List of matching DataReference objects
        """
        matches = []
        
        for data_type in DataType:
            for data_ref in self.data_inventory[data_type].values():
                # Check name pattern
                if name_pattern and name_pattern.lower() not in data_ref.name.lower():
                    continue
                
                # Check description pattern
                if description_pattern and description_pattern.lower() not in data_ref.description.lower():
                    continue
                
                # Check MIME type
                if mime_type and data_ref.mime_type != mime_type:
                    continue
                
                # Check size constraints
                if min_size and data_ref.size_bytes < min_size:
                    continue
                if max_size and data_ref.size_bytes > max_size:
                    continue
                
                matches.append(data_ref)
        
        logger.info(f"Found {len(matches)} data references matching search criteria")
        return matches
    
    def get_data_lineage(self, data_id: str) -> Dict[str, Any]:
        """
        Get complete data lineage for a specific data reference.
        
        Args:
            data_id: Data reference ID
            
        Returns:
            Dictionary containing lineage information
        """
        # Find the data reference
        target_ref = None
        for data_type in DataType:
            if data_id in self.data_inventory[data_type]:
                target_ref = self.data_inventory[data_type][data_id]
                break
        
        if not target_ref:
            logger.warning(f"Data reference {data_id} not found for lineage tracking")
            return {}
        
        lineage = {
            'target_data': target_ref.to_dict(),
            'parent_references': [],
            'transformation_chain': target_ref.provenance_info.transformation_history.copy(),
            'generation_step': target_ref.generated_by_step_id
        }
        
        # Find parent references
        for parent_id in target_ref.provenance_info.parent_data_ids:
            for data_type in DataType:
                if parent_id in self.data_inventory[data_type]:
                    lineage['parent_references'].append(
                        self.data_inventory[data_type][parent_id].to_dict()
                    )
        
        return lineage
    
    def validate_all_data_references(self) -> Dict[str, Any]:
        """
        Validate all data references in the inventory.
        
        Returns:
            Validation report with statistics
        """
        validation_report = {
            'total_references': 0,
            'valid_references': 0,
            'invalid_references': 0,
            'missing_files': [],
            'checksum_mismatches': [],
            'validation_timestamp': datetime.utcnow().isoformat() + "Z"
        }
        
        for data_type in DataType:
            for data_id, data_ref in self.data_inventory[data_type].items():
                validation_report['total_references'] += 1
                
                # Check if file exists
                if not data_ref.validate_file_exists():
                    validation_report['invalid_references'] += 1
                    validation_report['missing_files'].append({
                        'data_id': data_id,
                        'name': data_ref.name,
                        'path': data_ref.path,
                        'data_type': data_type.value
                    })
                    continue
                
                # Verify checksum if provided
                if data_ref.checksum_sha256:
                    current_checksum = data_ref._calculate_checksum()
                    if current_checksum != data_ref.checksum_sha256:
                        validation_report['invalid_references'] += 1
                        validation_report['checksum_mismatches'].append({
                            'data_id': data_id,
                            'name': data_ref.name,
                            'expected_checksum': data_ref.checksum_sha256,
                            'actual_checksum': current_checksum
                        })
                        continue
                
                validation_report['valid_references'] += 1
        
        logger.info(f"Data validation complete: {validation_report['valid_references']}/{validation_report['total_references']} valid")
        return validation_report
    
    def get_inventory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive inventory statistics."""
        stats = {
            'total_data_references': 0,
            'data_types': {},
            'total_size_bytes': 0,
            'total_record_count': 0,
            'mime_types': {},
            'generation_steps': {},
            'creation_timestamp': self.creation_timestamp,
            'last_update_timestamp': self.last_update_timestamp
        }
        
        for data_type in DataType:
            type_count = len(self.data_inventory[data_type])
            stats['data_types'][data_type.value] = type_count
            stats['total_data_references'] += type_count
            
            for data_ref in self.data_inventory[data_type].values():
                stats['total_size_bytes'] += data_ref.size_bytes
                stats['total_record_count'] += data_ref.record_count
                
                # Count MIME types
                mime_type = data_ref.mime_type or 'unknown'
                stats['mime_types'][mime_type] = stats['mime_types'].get(mime_type, 0) + 1
                
                # Count generation steps
                step_id = data_ref.generated_by_step_id or 'unknown'
                stats['generation_steps'][step_id] = stats['generation_steps'].get(step_id, 0) + 1
        
        return stats
    
    def export_inventory_to_dict(self) -> Dict[str, Any]:
        """Export complete inventory to dictionary format."""
        inventory_dict = {}
        
        for data_type in DataType:
            type_dict = {}
            for data_id, data_ref in self.data_inventory[data_type].items():
                type_dict[data_id] = data_ref.to_dict()
            inventory_dict[data_type.value] = type_dict
        
        return {
            'workflow_id': self.workflow_id,
            'creation_timestamp': self.creation_timestamp,
            'last_update_timestamp': self.last_update_timestamp,
            'data_inventory': inventory_dict,
            'statistics': self.get_inventory_statistics()
        }
    
    def import_inventory_from_dict(self, data: Dict[str, Any]) -> None:
        """Import inventory from dictionary format."""
        inventory_data = data.get('data_inventory', {})
        
        for data_type_str, type_dict in inventory_data.items():
            try:
                data_type = DataType(data_type_str)
                for data_id, ref_dict in type_dict.items():
                    # Reconstruct provenance info
                    provenance_data = ref_dict.get('provenance_info', {})
                    provenance_info = ProvenanceInfo(**provenance_data)
                    
                    # Remove nested dicts from ref_dict for DataReference creation
                    ref_data = ref_dict.copy()
                    ref_data.pop('provenance_info', None)
                    ref_data.pop('file_stats', None)
                    ref_data['provenance_info'] = provenance_info
                    
                    data_ref = DataReference(**ref_data)
                    self.data_inventory[data_type][data_id] = data_ref
                    
            except ValueError as e:
                logger.warning(f"Unknown data type during import: {data_type_str} - {e}")
        
        logger.info(f"Imported data inventory for workflow {self.workflow_id}")
    
    def cleanup_missing_references(self) -> int:
        """
        Remove data references for files that no longer exist.
        
        Returns:
            Number of references removed
        """
        removed_count = 0
        
        for data_type in DataType:
            to_remove = []
            for data_id, data_ref in self.data_inventory[data_type].items():
                if not data_ref.validate_file_exists():
                    to_remove.append(data_id)
            
            for data_id in to_remove:
                del self.data_inventory[data_type][data_id]
                removed_count += 1
                logger.info(f"Removed missing data reference: {data_id}")
        
        if removed_count > 0:
            self.last_update_timestamp = datetime.utcnow().isoformat() + "Z"
        
        return removed_count
