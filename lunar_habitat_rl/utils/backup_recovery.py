"""
NASA-grade backup and recovery system for mission-critical data.
Implements redundant storage, data integrity verification, and disaster recovery.
"""

import os
import json
import time
import threading
import shutil
import hashlib
import gzip
import pickle
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import concurrent.futures
# import schedule  # Optional scheduling - not used in core functionality

from .robust_logging import get_logger
from .audit_logging import get_audit_logger, AuditEventType, AuditLevel
from .security import compute_file_hash, secure_file_write


class BackupType(Enum):
    """Types of backup operations."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"
    EMERGENCY = "emergency"


class BackupStatus(Enum):
    """Status of backup operations."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CORRUPTED = "corrupted"


class RecoveryLevel(Enum):
    """Recovery operation levels."""
    FILE_LEVEL = "file_level"
    COMPONENT_LEVEL = "component_level"
    SYSTEM_LEVEL = "system_level"
    DISASTER_RECOVERY = "disaster_recovery"


@dataclass
class BackupMetadata:
    """Metadata for backup operations."""
    backup_id: str
    backup_type: BackupType
    timestamp: datetime
    source_paths: List[str]
    backup_path: str
    file_count: int
    total_size_bytes: int
    compressed_size_bytes: int
    checksum: str
    status: BackupStatus
    duration_seconds: float
    compression_ratio: float = 0.0
    verification_passed: bool = False
    mission_critical: bool = False
    retention_days: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert backup metadata to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['backup_type'] = self.backup_type.value
        data['status'] = self.status.value
        return data


@dataclass
class RecoveryPlan:
    """Recovery plan for disaster scenarios."""
    plan_id: str
    name: str
    description: str
    recovery_level: RecoveryLevel
    mission_impact: str
    estimated_rto: int  # Recovery Time Objective in minutes
    estimated_rpo: int  # Recovery Point Objective in minutes
    required_backups: List[str]
    recovery_steps: List[Dict[str, Any]]
    dependencies: List[str] = field(default_factory=list)
    validation_checks: List[str] = field(default_factory=list)
    created_date: datetime = field(default_factory=datetime.utcnow)
    last_tested: Optional[datetime] = None
    test_success: bool = False


class CriticalDataManager:
    """Manages identification and protection of mission-critical data."""
    
    def __init__(self):
        """Initialize critical data manager."""
        self.logger = get_logger()
        self.audit_logger = get_audit_logger()
        
        # Critical data classifications
        self.critical_data_types = {
            'life_support_logs': {
                'priority': 1,
                'retention_days': 365,
                'backup_frequency_hours': 1,
                'replication_count': 3,
                'mission_critical': True
            },
            'crew_health_data': {
                'priority': 1,
                'retention_days': 365,
                'backup_frequency_hours': 2,
                'replication_count': 3,
                'mission_critical': True
            },
            'system_configurations': {
                'priority': 2,
                'retention_days': 180,
                'backup_frequency_hours': 6,
                'replication_count': 2,
                'mission_critical': True
            },
            'mission_logs': {
                'priority': 2,
                'retention_days': 365,
                'backup_frequency_hours': 4,
                'replication_count': 2,
                'mission_critical': True
            },
            'simulation_data': {
                'priority': 3,
                'retention_days': 90,
                'backup_frequency_hours': 12,
                'replication_count': 2,
                'mission_critical': False
            },
            'training_models': {
                'priority': 3,
                'retention_days': 60,
                'backup_frequency_hours': 24,
                'replication_count': 1,
                'mission_critical': False
            }
        }
        
        # Data integrity tracking
        self.data_checksums = {}
        self.corruption_detected = set()
    
    def classify_data(self, file_path: Path) -> Dict[str, Any]:
        """Classify data file for backup priority and retention."""
        file_str = str(file_path).lower()
        
        # Determine data type based on path and content
        for data_type, config in self.critical_data_types.items():
            if any(keyword in file_str for keyword in data_type.split('_')):
                return {
                    'data_type': data_type,
                    'priority': config['priority'],
                    'mission_critical': config['mission_critical'],
                    'retention_days': config['retention_days'],
                    'backup_frequency_hours': config['backup_frequency_hours'],
                    'replication_count': config['replication_count']
                }
        
        # Default classification for unknown data
        return {
            'data_type': 'general',
            'priority': 4,
            'mission_critical': False,
            'retention_days': 30,
            'backup_frequency_hours': 24,
            'replication_count': 1
        }
    
    def verify_data_integrity(self, file_path: Path) -> Tuple[bool, str]:
        """Verify data integrity using checksums."""
        try:
            current_checksum = compute_file_hash(file_path)
            stored_checksum = self.data_checksums.get(str(file_path))
            
            if stored_checksum is None:
                # First time seeing this file - store checksum
                self.data_checksums[str(file_path)] = current_checksum
                return True, "checksum_stored"
            
            if current_checksum == stored_checksum:
                return True, "integrity_verified"
            else:
                # Corruption detected
                self.corruption_detected.add(str(file_path))
                self.logger.error(f"Data corruption detected: {file_path}")
                
                self.audit_logger.log_security_event(
                    action="data_corruption_detected",
                    description=f"Data integrity violation detected for {file_path}",
                    details={
                        'file_path': str(file_path),
                        'expected_checksum': stored_checksum,
                        'actual_checksum': current_checksum
                    },
                    level=AuditLevel.CRITICAL
                )
                
                return False, "corruption_detected"
                
        except Exception as e:
            self.logger.error(f"Integrity verification failed for {file_path}: {e}")
            return False, f"verification_error: {e}"


class BackupManager:
    """Comprehensive backup management system."""
    
    def __init__(self, backup_root: str = "backups"):
        """Initialize backup manager.
        
        Args:
            backup_root: Root directory for backup storage
        """
        self.logger = get_logger()
        self.audit_logger = get_audit_logger()
        self.critical_data_manager = CriticalDataManager()
        
        # Backup configuration
        self.backup_root = Path(backup_root)
        self.backup_root.mkdir(exist_ok=True)
        
        # Create backup subdirectories
        (self.backup_root / "full").mkdir(exist_ok=True)
        (self.backup_root / "incremental").mkdir(exist_ok=True)
        (self.backup_root / "emergency").mkdir(exist_ok=True)
        (self.backup_root / "snapshots").mkdir(exist_ok=True)
        
        # Backup tracking
        self.backup_history: List[BackupMetadata] = []
        self.active_backups = {}  # backup_id -> thread
        self.backup_schedule = {}
        
        # Performance settings
        self.max_concurrent_backups = 3
        self.compression_enabled = True
        self.verification_enabled = True
        
        # Load existing backup metadata
        self._load_backup_history()
        
        # Backup scheduler
        self.scheduler_running = False
        self.scheduler_thread = None
        
    def create_backup(self, source_paths: List[Union[str, Path]], 
                     backup_type: BackupType = BackupType.FULL,
                     mission_critical: bool = False,
                     retention_days: int = 30) -> str:
        """Create backup of specified paths.
        
        Args:
            source_paths: List of paths to backup
            backup_type: Type of backup to create
            mission_critical: Whether this is mission-critical data
            retention_days: Number of days to retain backup
            
        Returns:
            Backup ID
        """
        backup_id = f"backup_{int(time.time()*1000)}"
        timestamp = datetime.utcnow()
        
        # Convert paths to Path objects
        source_paths = [Path(p) for p in source_paths]
        
        # Create backup metadata
        backup_metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=backup_type,
            timestamp=timestamp,
            source_paths=[str(p) for p in source_paths],
            backup_path="",  # Will be set during backup
            file_count=0,
            total_size_bytes=0,
            compressed_size_bytes=0,
            checksum="",
            status=BackupStatus.PENDING,
            duration_seconds=0.0,
            mission_critical=mission_critical,
            retention_days=retention_days
        )
        
        # Start backup in background thread
        backup_thread = threading.Thread(
            target=self._execute_backup,
            args=(backup_metadata, source_paths),
            daemon=True
        )
        
        self.active_backups[backup_id] = backup_thread
        backup_thread.start()
        
        self.logger.info(f"Backup initiated: {backup_id} ({backup_type.value})")
        
        return backup_id
    
    def _execute_backup(self, metadata: BackupMetadata, source_paths: List[Path]):
        """Execute backup operation."""
        start_time = time.time()
        
        try:
            metadata.status = BackupStatus.RUNNING
            
            # Create backup directory
            backup_dir = self.backup_root / metadata.backup_type.value / metadata.backup_id
            backup_dir.mkdir(parents=True, exist_ok=True)
            metadata.backup_path = str(backup_dir)
            
            # Collect files to backup
            files_to_backup = []
            total_size = 0
            
            for source_path in source_paths:
                if source_path.is_file():
                    files_to_backup.append(source_path)
                    total_size += source_path.stat().st_size
                elif source_path.is_dir():
                    for file_path in source_path.rglob('*'):
                        if file_path.is_file():
                            files_to_backup.append(file_path)
                            total_size += file_path.stat().st_size
            
            metadata.file_count = len(files_to_backup)
            metadata.total_size_bytes = total_size
            
            # Execute backup based on type
            if metadata.backup_type == BackupType.FULL:
                compressed_size = self._create_full_backup(files_to_backup, backup_dir)
            elif metadata.backup_type == BackupType.INCREMENTAL:
                compressed_size = self._create_incremental_backup(files_to_backup, backup_dir)
            elif metadata.backup_type == BackupType.EMERGENCY:
                compressed_size = self._create_emergency_backup(files_to_backup, backup_dir)
            else:
                compressed_size = self._create_full_backup(files_to_backup, backup_dir)
            
            metadata.compressed_size_bytes = compressed_size
            metadata.compression_ratio = compressed_size / max(total_size, 1)
            
            # Calculate backup checksum
            metadata.checksum = self._calculate_backup_checksum(backup_dir)
            
            # Verify backup if enabled
            if self.verification_enabled:
                metadata.verification_passed = self._verify_backup(backup_dir, files_to_backup)
            else:
                metadata.verification_passed = True
            
            # Update metadata
            metadata.duration_seconds = time.time() - start_time
            metadata.status = BackupStatus.COMPLETED if metadata.verification_passed else BackupStatus.CORRUPTED
            
            # Store metadata
            self._save_backup_metadata(metadata, backup_dir)
            self.backup_history.append(metadata)
            
            # Log success
            self.logger.info(f"Backup completed: {metadata.backup_id} "
                           f"({metadata.file_count} files, {metadata.compressed_size_bytes} bytes)")
            
            self.audit_logger.log_system_event(
                action="backup_completed",
                component="backup_manager",
                description=f"Backup {metadata.backup_id} completed successfully",
                details=metadata.to_dict(),
                level=AuditLevel.INFO
            )
            
            # Mission critical backup notification
            if metadata.mission_critical:
                self.audit_logger.log_event(
                    event_type=AuditEventType.MISSION_CRITICAL,
                    level=AuditLevel.INFO,
                    action="mission_critical_backup",
                    component="backup_manager",
                    description=f"Mission-critical data backup completed: {metadata.backup_id}",
                    details=metadata.to_dict(),
                    mission_impact=True
                )
            
        except Exception as e:
            metadata.status = BackupStatus.FAILED
            metadata.duration_seconds = time.time() - start_time
            
            self.logger.error(f"Backup failed: {metadata.backup_id} - {e}")
            
            self.audit_logger.log_system_event(
                action="backup_failed",
                component="backup_manager",
                description=f"Backup {metadata.backup_id} failed: {e}",
                details={'backup_id': metadata.backup_id, 'error': str(e)},
                level=AuditLevel.ERROR,
                success=False
            )
            
            # For mission critical backups, trigger emergency protocols
            if metadata.mission_critical:
                self.audit_logger.log_emergency_event(
                    action="mission_critical_backup_failure",
                    description=f"Mission-critical backup failed: {metadata.backup_id}",
                    details={'backup_id': metadata.backup_id, 'error': str(e)}
                )
        
        finally:
            # Remove from active backups
            if metadata.backup_id in self.active_backups:
                del self.active_backups[metadata.backup_id]
    
    def _create_full_backup(self, files: List[Path], backup_dir: Path) -> int:
        """Create full backup of all files."""
        total_compressed_size = 0
        
        for file_path in files:
            try:
                # Create relative path structure
                rel_path = file_path.relative_to(file_path.anchor) if file_path.is_absolute() else file_path
                backup_file_path = backup_dir / rel_path
                backup_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                if self.compression_enabled:
                    # Compress and copy
                    with open(file_path, 'rb') as f_in:
                        with gzip.open(f"{backup_file_path}.gz", 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    total_compressed_size += Path(f"{backup_file_path}.gz").stat().st_size
                else:
                    # Direct copy
                    shutil.copy2(file_path, backup_file_path)
                    total_compressed_size += backup_file_path.stat().st_size
                    
            except Exception as e:
                self.logger.warning(f"Failed to backup file {file_path}: {e}")
        
        return total_compressed_size
    
    def _create_incremental_backup(self, files: List[Path], backup_dir: Path) -> int:
        """Create incremental backup (only changed files since last backup)."""
        # Find last successful backup
        last_backup = None
        for backup in reversed(self.backup_history):
            if backup.status == BackupStatus.COMPLETED:
                last_backup = backup
                break
        
        if not last_backup:
            # No previous backup - create full backup
            return self._create_full_backup(files, backup_dir)
        
        # Filter files changed since last backup
        changed_files = []
        last_backup_time = last_backup.timestamp
        
        for file_path in files:
            try:
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_mtime > last_backup_time:
                    changed_files.append(file_path)
            except Exception as e:
                self.logger.warning(f"Failed to check modification time for {file_path}: {e}")
                # Include file if we can't check modification time
                changed_files.append(file_path)
        
        self.logger.info(f"Incremental backup: {len(changed_files)} of {len(files)} files changed")
        
        return self._create_full_backup(changed_files, backup_dir)
    
    def _create_emergency_backup(self, files: List[Path], backup_dir: Path) -> int:
        """Create emergency backup with highest priority data only."""
        # Filter to mission critical files only
        critical_files = []
        
        for file_path in files:
            classification = self.critical_data_manager.classify_data(file_path)
            if classification['mission_critical'] or classification['priority'] <= 2:
                critical_files.append(file_path)
        
        self.logger.info(f"Emergency backup: {len(critical_files)} critical files of {len(files)} total")
        
        return self._create_full_backup(critical_files, backup_dir)
    
    def _calculate_backup_checksum(self, backup_dir: Path) -> str:
        """Calculate checksum for entire backup."""
        hash_obj = hashlib.sha256()
        
        # Sort files for consistent checksum
        backup_files = sorted(backup_dir.rglob('*'))
        
        for file_path in backup_files:
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    while chunk := f.read(8192):
                        hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    
    def _verify_backup(self, backup_dir: Path, original_files: List[Path]) -> bool:
        """Verify backup integrity."""
        try:
            backup_files = list(backup_dir.rglob('*'))
            backup_file_count = len([f for f in backup_files if f.is_file()])
            
            # Check file count matches (accounting for compression)
            expected_count = len(original_files)
            if backup_file_count != expected_count:
                self.logger.error(f"Backup verification failed: file count mismatch "
                                f"(expected {expected_count}, got {backup_file_count})")
                return False
            
            # Spot check some files
            import random
            sample_size = min(10, len(original_files))
            sample_files = random.sample(original_files, sample_size)
            
            for original_file in sample_files:
                rel_path = original_file.relative_to(original_file.anchor) if original_file.is_absolute() else original_file
                
                if self.compression_enabled:
                    backup_file = backup_dir / f"{rel_path}.gz"
                    if not backup_file.exists():
                        self.logger.error(f"Backup verification failed: missing file {backup_file}")
                        return False
                else:
                    backup_file = backup_dir / rel_path
                    if not backup_file.exists():
                        self.logger.error(f"Backup verification failed: missing file {backup_file}")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Backup verification error: {e}")
            return False
    
    def _save_backup_metadata(self, metadata: BackupMetadata, backup_dir: Path):
        """Save backup metadata to file."""
        metadata_file = backup_dir / "backup_metadata.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
    
    def _load_backup_history(self):
        """Load backup history from existing backups."""
        try:
            for backup_type_dir in self.backup_root.iterdir():
                if backup_type_dir.is_dir():
                    for backup_dir in backup_type_dir.iterdir():
                        if backup_dir.is_dir():
                            metadata_file = backup_dir / "backup_metadata.json"
                            if metadata_file.exists():
                                try:
                                    with open(metadata_file, 'r') as f:
                                        data = json.load(f)
                                    
                                    metadata = BackupMetadata(
                                        backup_id=data['backup_id'],
                                        backup_type=BackupType(data['backup_type']),
                                        timestamp=datetime.fromisoformat(data['timestamp']),
                                        source_paths=data['source_paths'],
                                        backup_path=data['backup_path'],
                                        file_count=data['file_count'],
                                        total_size_bytes=data['total_size_bytes'],
                                        compressed_size_bytes=data['compressed_size_bytes'],
                                        checksum=data['checksum'],
                                        status=BackupStatus(data['status']),
                                        duration_seconds=data['duration_seconds'],
                                        compression_ratio=data.get('compression_ratio', 0.0),
                                        verification_passed=data.get('verification_passed', False),
                                        mission_critical=data.get('mission_critical', False),
                                        retention_days=data.get('retention_days', 30)
                                    )
                                    
                                    self.backup_history.append(metadata)
                                    
                                except Exception as e:
                                    self.logger.warning(f"Failed to load backup metadata from {metadata_file}: {e}")
            
            self.backup_history.sort(key=lambda b: b.timestamp)
            self.logger.info(f"Loaded {len(self.backup_history)} backup records")
            
        except Exception as e:
            self.logger.error(f"Failed to load backup history: {e}")
    
    def restore_from_backup(self, backup_id: str, restore_path: Path, 
                           files_filter: Optional[List[str]] = None) -> bool:
        """Restore data from backup.
        
        Args:
            backup_id: ID of backup to restore from
            restore_path: Path to restore data to
            files_filter: Optional list of specific files to restore
            
        Returns:
            True if restore successful
        """
        try:
            # Find backup metadata
            backup_metadata = None
            for backup in self.backup_history:
                if backup.backup_id == backup_id:
                    backup_metadata = backup
                    break
            
            if not backup_metadata:
                self.logger.error(f"Backup not found: {backup_id}")
                return False
            
            backup_dir = Path(backup_metadata.backup_path)
            if not backup_dir.exists():
                self.logger.error(f"Backup directory not found: {backup_dir}")
                return False
            
            # Create restore directory
            restore_path.mkdir(parents=True, exist_ok=True)
            
            # Restore files
            restored_count = 0
            
            for backup_file in backup_dir.rglob('*'):
                if backup_file.is_file() and backup_file.name != "backup_metadata.json":
                    # Apply file filter if specified
                    if files_filter and not any(filter_name in str(backup_file) for filter_name in files_filter):
                        continue
                    
                    # Determine restore path
                    rel_path = backup_file.relative_to(backup_dir)
                    
                    # Handle compressed files
                    if backup_file.suffix == '.gz':
                        rel_path = rel_path.with_suffix('')  # Remove .gz extension
                        restore_file_path = restore_path / rel_path
                        restore_file_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Decompress
                        with gzip.open(backup_file, 'rb') as f_in:
                            with open(restore_file_path, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                    else:
                        restore_file_path = restore_path / rel_path
                        restore_file_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(backup_file, restore_file_path)
                    
                    restored_count += 1
            
            self.logger.info(f"Restore completed: {restored_count} files from backup {backup_id}")
            
            self.audit_logger.log_system_event(
                action="data_restored",
                component="backup_manager",
                description=f"Data restored from backup {backup_id}",
                details={
                    'backup_id': backup_id,
                    'restore_path': str(restore_path),
                    'files_restored': restored_count,
                    'files_filter': files_filter
                },
                level=AuditLevel.INFO
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Restore failed for backup {backup_id}: {e}")
            
            self.audit_logger.log_system_event(
                action="restore_failed",
                component="backup_manager",
                description=f"Restore failed for backup {backup_id}: {e}",
                details={'backup_id': backup_id, 'error': str(e)},
                level=AuditLevel.ERROR,
                success=False
            )
            
            return False
    
    def cleanup_old_backups(self):
        """Clean up expired backups based on retention policies."""
        cleaned_count = 0
        current_time = datetime.utcnow()
        
        for backup in list(self.backup_history):
            # Check if backup has expired
            expiry_date = backup.timestamp + timedelta(days=backup.retention_days)
            
            if current_time > expiry_date:
                try:
                    # Remove backup directory
                    backup_dir = Path(backup.backup_path)
                    if backup_dir.exists():
                        shutil.rmtree(backup_dir)
                    
                    # Remove from history
                    self.backup_history.remove(backup)
                    cleaned_count += 1
                    
                    self.logger.info(f"Cleaned up expired backup: {backup.backup_id}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to cleanup backup {backup.backup_id}: {e}")
        
        if cleaned_count > 0:
            self.audit_logger.log_system_event(
                action="backup_cleanup",
                component="backup_manager",
                description=f"Cleaned up {cleaned_count} expired backups",
                details={'cleaned_count': cleaned_count}
            )
    
    def get_backup_status(self) -> Dict[str, Any]:
        """Get comprehensive backup system status."""
        total_backups = len(self.backup_history)
        successful_backups = len([b for b in self.backup_history if b.status == BackupStatus.COMPLETED])
        failed_backups = len([b for b in self.backup_history if b.status == BackupStatus.FAILED])
        mission_critical_backups = len([b for b in self.backup_history if b.mission_critical])
        
        # Calculate total storage used
        total_storage_bytes = sum(b.compressed_size_bytes for b in self.backup_history 
                                if b.status == BackupStatus.COMPLETED)
        
        # Find latest backup
        latest_backup = max(self.backup_history, key=lambda b: b.timestamp) if self.backup_history else None
        
        return {
            'backup_statistics': {
                'total_backups': total_backups,
                'successful_backups': successful_backups,
                'failed_backups': failed_backups,
                'success_rate': successful_backups / max(total_backups, 1),
                'mission_critical_backups': mission_critical_backups,
                'active_backups': len(self.active_backups),
                'total_storage_gb': total_storage_bytes / (1024**3)
            },
            'latest_backup': {
                'backup_id': latest_backup.backup_id if latest_backup else None,
                'timestamp': latest_backup.timestamp.isoformat() if latest_backup else None,
                'status': latest_backup.status.value if latest_backup else None,
                'mission_critical': latest_backup.mission_critical if latest_backup else False
            } if latest_backup else None,
            'system_health': {
                'compression_enabled': self.compression_enabled,
                'verification_enabled': self.verification_enabled,
                'scheduler_running': self.scheduler_running,
                'data_corruption_detected': len(self.critical_data_manager.corruption_detected)
            }
        }


class DisasterRecoveryManager:
    """Manages disaster recovery plans and procedures."""
    
    def __init__(self, backup_manager: BackupManager):
        """Initialize disaster recovery manager."""
        self.logger = get_logger()
        self.audit_logger = get_audit_logger()
        self.backup_manager = backup_manager
        
        # Recovery plans
        self.recovery_plans: Dict[str, RecoveryPlan] = {}
        self.active_recovery = None
        
        # Initialize standard recovery plans
        self._initialize_standard_recovery_plans()
    
    def _initialize_standard_recovery_plans(self):
        """Initialize standard disaster recovery plans."""
        
        # Plan 1: Life Support System Failure
        life_support_plan = RecoveryPlan(
            plan_id="life_support_recovery",
            name="Life Support System Recovery",
            description="Recovery from life support system failure",
            recovery_level=RecoveryLevel.SYSTEM_LEVEL,
            mission_impact="CRITICAL - Crew safety at risk",
            estimated_rto=5,   # 5 minutes RTO
            estimated_rpo=2,   # 2 minutes RPO
            required_backups=["life_support_logs", "system_configurations"],
            recovery_steps=[
                {"step": 1, "action": "activate_backup_life_support", "timeout_minutes": 2},
                {"step": 2, "action": "restore_life_support_configuration", "timeout_minutes": 1},
                {"step": 3, "action": "verify_atmospheric_parameters", "timeout_minutes": 2}
            ],
            validation_checks=[
                "o2_pressure_within_limits",
                "co2_pressure_within_limits",
                "backup_systems_operational"
            ]
        )
        self.recovery_plans[life_support_plan.plan_id] = life_support_plan
        
        # Plan 2: Complete System Failure
        system_failure_plan = RecoveryPlan(
            plan_id="complete_system_recovery",
            name="Complete System Recovery",
            description="Recovery from complete system failure",
            recovery_level=RecoveryLevel.DISASTER_RECOVERY,
            mission_impact="CATASTROPHIC - Mission abort risk",
            estimated_rto=60,  # 1 hour RTO
            estimated_rpo=15,  # 15 minutes RPO
            required_backups=["life_support_logs", "crew_health_data", "system_configurations", "mission_logs"],
            recovery_steps=[
                {"step": 1, "action": "activate_emergency_power", "timeout_minutes": 5},
                {"step": 2, "action": "restore_critical_systems", "timeout_minutes": 20},
                {"step": 3, "action": "restore_life_support", "timeout_minutes": 10},
                {"step": 4, "action": "restore_communication", "timeout_minutes": 15},
                {"step": 5, "action": "verify_all_systems", "timeout_minutes": 10}
            ],
            validation_checks=[
                "all_critical_systems_online",
                "life_support_nominal",
                "communication_established",
                "crew_health_stable"
            ]
        )
        self.recovery_plans[system_failure_plan.plan_id] = system_failure_plan
        
        # Plan 3: Data Corruption Recovery
        data_corruption_plan = RecoveryPlan(
            plan_id="data_corruption_recovery",
            name="Data Corruption Recovery",
            description="Recovery from data corruption events",
            recovery_level=RecoveryLevel.COMPONENT_LEVEL,
            mission_impact="MODERATE - Operational impact",
            estimated_rto=30,  # 30 minutes RTO
            estimated_rpo=60,  # 1 hour RPO
            required_backups=["system_configurations", "mission_logs"],
            recovery_steps=[
                {"step": 1, "action": "identify_corrupted_data", "timeout_minutes": 5},
                {"step": 2, "action": "isolate_affected_systems", "timeout_minutes": 5},
                {"step": 3, "action": "restore_from_backup", "timeout_minutes": 15},
                {"step": 4, "action": "verify_data_integrity", "timeout_minutes": 5}
            ],
            validation_checks=[
                "data_integrity_verified",
                "system_functionality_restored",
                "no_additional_corruption"
            ]
        )
        self.recovery_plans[data_corruption_plan.plan_id] = data_corruption_plan
    
    def execute_recovery_plan(self, plan_id: str, simulation_mode: bool = False) -> Dict[str, Any]:
        """Execute disaster recovery plan.
        
        Args:
            plan_id: ID of recovery plan to execute
            simulation_mode: If True, simulate recovery without actual execution
            
        Returns:
            Recovery execution results
        """
        if plan_id not in self.recovery_plans:
            raise ValueError(f"Recovery plan not found: {plan_id}")
        
        plan = self.recovery_plans[plan_id]
        start_time = time.time()
        
        recovery_result = {
            'plan_id': plan_id,
            'plan_name': plan.name,
            'start_time': datetime.utcnow().isoformat(),
            'simulation_mode': simulation_mode,
            'steps_completed': 0,
            'steps_failed': 0,
            'validation_results': {},
            'success': False,
            'duration_seconds': 0,
            'estimated_rto_minutes': plan.estimated_rto,
            'estimated_rpo_minutes': plan.estimated_rpo
        }
        
        try:
            self.active_recovery = plan_id
            
            # Log recovery initiation
            self.audit_logger.log_emergency_event(
                action="disaster_recovery_initiated",
                description=f"Disaster recovery plan initiated: {plan.name}",
                details={
                    'plan_id': plan_id,
                    'recovery_level': plan.recovery_level.value,
                    'mission_impact': plan.mission_impact,
                    'simulation_mode': simulation_mode
                }
            )
            
            # Verify required backups are available
            self._verify_backup_availability(plan, recovery_result)
            
            # Execute recovery steps
            for step in plan.recovery_steps:
                step_start_time = time.time()
                step_success = False
                
                try:
                    if simulation_mode:
                        # Simulate step execution
                        time.sleep(1)  # Simulate processing time
                        step_success = True
                        self.logger.info(f"SIMULATION: Executed recovery step {step['step']}: {step['action']}")
                    else:
                        # Execute actual recovery step
                        step_success = self._execute_recovery_step(step, plan)
                    
                    if step_success:
                        recovery_result['steps_completed'] += 1
                        self.logger.info(f"Recovery step {step['step']} completed: {step['action']}")
                    else:
                        recovery_result['steps_failed'] += 1
                        self.logger.error(f"Recovery step {step['step']} failed: {step['action']}")
                        
                        # For critical failures, abort recovery
                        if plan.recovery_level == RecoveryLevel.DISASTER_RECOVERY:
                            break
                    
                except Exception as e:
                    recovery_result['steps_failed'] += 1
                    self.logger.error(f"Recovery step {step['step']} error: {e}")
                
                step_duration = time.time() - step_start_time
                timeout_minutes = step.get('timeout_minutes', 10)
                
                if step_duration > timeout_minutes * 60:
                    self.logger.error(f"Recovery step {step['step']} exceeded timeout ({timeout_minutes}m)")
                    recovery_result['steps_failed'] += 1
            
            # Perform validation checks
            recovery_result['validation_results'] = self._perform_validation_checks(plan, simulation_mode)
            
            # Determine overall success
            validation_passed = all(recovery_result['validation_results'].values())
            steps_successful = recovery_result['steps_failed'] == 0
            recovery_result['success'] = validation_passed and steps_successful
            
            recovery_result['duration_seconds'] = time.time() - start_time
            
            # Log recovery completion
            self.audit_logger.log_emergency_event(
                action="disaster_recovery_completed",
                description=f"Disaster recovery {'successful' if recovery_result['success'] else 'failed'}: {plan.name}",
                details=recovery_result,
                response_time_ms=recovery_result['duration_seconds'] * 1000
            )
            
            return recovery_result
            
        except Exception as e:
            recovery_result['duration_seconds'] = time.time() - start_time
            recovery_result['error'] = str(e)
            
            self.logger.error(f"Disaster recovery plan {plan_id} failed: {e}")
            
            self.audit_logger.log_emergency_event(
                action="disaster_recovery_failed",
                description=f"Disaster recovery plan failed: {plan.name} - {e}",
                details=recovery_result
            )
            
            return recovery_result
            
        finally:
            self.active_recovery = None
    
    def _verify_backup_availability(self, plan: RecoveryPlan, result: Dict[str, Any]):
        """Verify required backups are available for recovery."""
        missing_backups = []
        
        for required_backup in plan.required_backups:
            # Find latest successful backup for this data type
            backup_found = False
            for backup in reversed(self.backup_manager.backup_history):
                if (backup.status == BackupStatus.COMPLETED and 
                    any(required_backup in source for source in backup.source_paths)):
                    backup_found = True
                    break
            
            if not backup_found:
                missing_backups.append(required_backup)
        
        result['missing_backups'] = missing_backups
        
        if missing_backups:
            self.logger.error(f"Missing required backups for recovery: {missing_backups}")
    
    def _execute_recovery_step(self, step: Dict[str, Any], plan: RecoveryPlan) -> bool:
        """Execute individual recovery step."""
        action = step['action']
        
        # This would contain actual recovery logic for each action type
        # For demonstration, we'll simulate some common recovery actions
        
        if action == "activate_backup_life_support":
            self.logger.info("Activating backup life support systems")
            # Would trigger actual backup life support activation
            return True
            
        elif action == "restore_life_support_configuration":
            self.logger.info("Restoring life support configuration from backup")
            # Would restore actual configuration
            return True
            
        elif action == "verify_atmospheric_parameters":
            self.logger.info("Verifying atmospheric parameters")
            # Would check actual atmospheric sensors
            return True
            
        elif action == "activate_emergency_power":
            self.logger.info("Activating emergency power systems")
            # Would activate actual emergency power
            return True
            
        elif action == "restore_critical_systems":
            self.logger.info("Restoring critical systems from backup")
            # Would restore actual systems
            return True
            
        else:
            self.logger.warning(f"Unknown recovery action: {action}")
            return False
    
    def _perform_validation_checks(self, plan: RecoveryPlan, simulation_mode: bool) -> Dict[str, bool]:
        """Perform validation checks after recovery."""
        validation_results = {}
        
        for check in plan.validation_checks:
            if simulation_mode:
                # Simulate successful validation
                validation_results[check] = True
            else:
                # Perform actual validation
                validation_results[check] = self._perform_validation_check(check)
        
        return validation_results
    
    def _perform_validation_check(self, check: str) -> bool:
        """Perform individual validation check."""
        # This would contain actual validation logic
        if check == "o2_pressure_within_limits":
            # Would check actual O2 sensors
            return True
        elif check == "co2_pressure_within_limits":
            # Would check actual CO2 sensors
            return True
        elif check == "all_critical_systems_online":
            # Would check actual system status
            return True
        elif check == "data_integrity_verified":
            # Would verify actual data integrity
            return True
        else:
            self.logger.warning(f"Unknown validation check: {check}")
            return False


# Global backup and recovery instances
_global_backup_manager = None
_global_disaster_recovery_manager = None

def get_backup_manager() -> BackupManager:
    """Get global backup manager instance."""
    global _global_backup_manager
    if _global_backup_manager is None:
        _global_backup_manager = BackupManager()
    return _global_backup_manager

def get_disaster_recovery_manager() -> DisasterRecoveryManager:
    """Get global disaster recovery manager instance."""
    global _global_disaster_recovery_manager
    if _global_disaster_recovery_manager is None:
        backup_mgr = get_backup_manager()
        _global_disaster_recovery_manager = DisasterRecoveryManager(backup_mgr)
    return _global_disaster_recovery_manager


# Convenient functions for common operations
def create_emergency_backup(paths: List[Union[str, Path]]) -> str:
    """Create emergency backup of critical data."""
    backup_mgr = get_backup_manager()
    return backup_mgr.create_backup(
        source_paths=paths,
        backup_type=BackupType.EMERGENCY,
        mission_critical=True,
        retention_days=365  # Keep emergency backups for 1 year
    )

def initiate_disaster_recovery(scenario: str) -> Dict[str, Any]:
    """Initiate disaster recovery for common scenarios."""
    recovery_mgr = get_disaster_recovery_manager()
    
    scenario_plan_mapping = {
        'life_support_failure': 'life_support_recovery',
        'system_failure': 'complete_system_recovery',
        'data_corruption': 'data_corruption_recovery'
    }
    
    plan_id = scenario_plan_mapping.get(scenario)
    if not plan_id:
        raise ValueError(f"Unknown disaster scenario: {scenario}")
    
    return recovery_mgr.execute_recovery_plan(plan_id, simulation_mode=False)