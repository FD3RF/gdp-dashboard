"""
File Manager Agent
==================

Manages file operations with safety checks and version tracking.
Prevents accidental overwrites and maintains file history.
"""

import os
import shutil
import difflib
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import json

from core.base import BaseModule
from core.memory import MemoryStore, MemoryType


@dataclass
class FileVersion:
    """Tracks a file version."""
    path: str
    content_hash: str
    timestamp: str
    size: int
    action: str  # 'create', 'modify', 'delete'


class FileManager(BaseModule):
    """
    Safe file management for AI agents.
    
    Features:
    - Safe file operations with backup
    - Diff-based modifications (avoid full rewrites)
    - File history tracking
    - Rollback capability
    - Project structure management
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('file_manager', config)
        
        self.workspace_path = Path(self.config.get('workspace', '.'))
        self.backup_dir = self.workspace_path / '.file_backups'
        self.history_file = self.backup_dir / 'file_history.json'
        
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.memory = MemoryStore()
        self._history: List[FileVersion] = []
        self._load_history()
    
    async def initialize(self) -> bool:
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        self._running = True
        return True
    
    async def stop(self) -> bool:
        self._save_history()
        self._running = False
        return True
    
    def _load_history(self):
        """Load file history from disk."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self._history = [FileVersion(**v) for v in data]
            except Exception as e:
                self.logger.warning(f"Failed to load history: {e}")
    
    def _save_history(self):
        """Save file history to disk."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump([vars(v) for v in self._history], f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save history: {e}")
    
    def _hash_content(self, content: str) -> str:
        """Generate hash for content."""
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _backup_file(self, file_path: Path) -> Optional[str]:
        """Create backup of existing file."""
        if not file_path.exists():
            return None
        
        content = file_path.read_text()
        content_hash = self._hash_content(content)
        backup_name = f"{file_path.name}.{content_hash}.bak"
        backup_path = self.backup_dir / backup_name
        
        shutil.copy2(file_path, backup_path)
        self.logger.info(f"Backed up: {file_path} -> {backup_path}")
        
        return str(backup_path)
    
    # ==================== Core File Operations ====================
    
    async def create_file(self, file_path: str, content: str, 
                          description: str = None) -> Dict[str, Any]:
        """
        Safely create a new file.
        
        Args:
            file_path: Relative path to create
            content: File content
            description: What this file is for
            
        Returns:
            Result dictionary with status
        """
        full_path = self.workspace_path / file_path
        
        # Check if file already exists
        if full_path.exists():
            return {
                'success': False,
                'error': f"File already exists: {file_path}",
                'suggestion': "Use modify_file() to update existing files"
            }
        
        # Create parent directories
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        full_path.write_text(content)
        
        # Record in history
        version = FileVersion(
            path=file_path,
            content_hash=self._hash_content(content),
            timestamp=datetime.now().isoformat(),
            size=len(content),
            action='create'
        )
        self._history.append(version)
        self._save_history()
        
        # Update memory
        self.memory.remember(
            MemoryType.CODE_SNIPPET,
            {'action': 'create', 'path': file_path, 'description': description},
            importance=0.7
        )
        
        self.logger.info(f"Created: {file_path}")
        
        return {
            'success': True,
            'path': file_path,
            'size': len(content),
            'hash': version.content_hash
        }
    
    async def read_file(self, file_path: str) -> Dict[str, Any]:
        """
        Read file content.
        
        Args:
            file_path: Relative path to read
            
        Returns:
            Result with content and metadata
        """
        full_path = self.workspace_path / file_path
        
        if not full_path.exists():
            return {
                'success': False,
                'error': f"File not found: {file_path}"
            }
        
        content = full_path.read_text()
        
        return {
            'success': True,
            'path': file_path,
            'content': content,
            'size': len(content),
            'hash': self._hash_content(content),
            'modified': datetime.fromtimestamp(full_path.stat().st_mtime).isoformat()
        }
    
    async def modify_file(self, file_path: str, old_content: str, 
                          new_content: str, reason: str = None) -> Dict[str, Any]:
        """
        Safely modify a file using diff-based approach.
        
        Args:
            file_path: Relative path to modify
            old_content: Content to find and replace (must be unique)
            new_content: New content to insert
            reason: Why this modification is needed
            
        Returns:
            Result dictionary
        """
        full_path = self.workspace_path / file_path
        
        if not full_path.exists():
            return {
                'success': False,
                'error': f"File not found: {file_path}"
            }
        
        current_content = full_path.read_text()
        
        # Check if old_content exists
        if old_content not in current_content:
            return {
                'success': False,
                'error': "Content to replace not found in file",
                'hint': "Ensure old_content matches exactly what's in the file"
            }
        
        # Check for uniqueness
        occurrences = current_content.count(old_content)
        if occurrences > 1:
            return {
                'success': False,
                'error': f"Content appears {occurrences} times - not unique",
                'hint': "Provide more context to make the match unique"
            }
        
        # Create backup
        backup_path = self._backup_file(full_path)
        
        # Apply modification
        modified_content = current_content.replace(old_content, new_content)
        full_path.write_text(modified_content)
        
        # Record in history
        version = FileVersion(
            path=file_path,
            content_hash=self._hash_content(modified_content),
            timestamp=datetime.now().isoformat(),
            size=len(modified_content),
            action='modify'
        )
        self._history.append(version)
        self._save_history()
        
        # Generate diff for logging
        diff = ''.join(difflib.unified_diff(
            current_content.splitlines(keepends=True),
            modified_content.splitlines(keepends=True),
            fromfile=f"{file_path}.old",
            tofile=file_path
        ))
        
        self.memory.remember(
            MemoryType.CODE_SNIPPET,
            {'action': 'modify', 'path': file_path, 'reason': reason},
            importance=0.7
        )
        
        self.logger.info(f"Modified: {file_path}")
        
        return {
            'success': True,
            'path': file_path,
            'backup': backup_path,
            'diff': diff,
            'size_change': len(modified_content) - len(current_content)
        }
    
    async def apply_patch(self, file_path: str, patch: str) -> Dict[str, Any]:
        """
        Apply a unified diff patch to a file.
        
        Args:
            file_path: Relative path to patch
            patch: Unified diff format patch
            
        Returns:
            Result dictionary
        """
        full_path = self.workspace_path / file_path
        
        if not full_path.exists():
            return {
                'success': False,
                'error': f"File not found: {file_path}"
            }
        
        # Create backup
        backup_path = self._backup_file(full_path)
        
        try:
            # Parse and apply patch
            current_content = full_path.read_text()
            lines = current_content.splitlines(keepends=True)
            
            # Simple patch application (basic unified diff)
            patch_lines = patch.splitlines()
            result_lines = lines.copy()
            line_offset = 0
            
            for line in patch_lines:
                if line.startswith('@@'):
                    # Parse hunk header
                    match = self._parse_hunk_header(line)
                    if match:
                        old_start, old_count, new_start, new_count = match
                        # Apply changes...
                elif line.startswith('-') and not line.startswith('---'):
                    # Remove line
                    pass
                elif line.startswith('+') and not line.startswith('+++'):
                    # Add line
                    pass
            
            # For safety, use a simpler approach
            # This is a simplified implementation
            self.logger.warning("Patch application not fully implemented - using line replacement")
            
            return {
                'success': False,
                'error': "Patch format not supported - use modify_file() instead",
                'suggestion': "Use modify_file() with old_content and new_content"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to apply patch: {str(e)}"
            }
    
    def _parse_hunk_header(self, line: str) -> Optional[Tuple[int, int, int, int]]:
        """Parse unified diff hunk header."""
        import re
        match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
        if match:
            return (
                int(match.group(1)),
                int(match.group(2) or 1),
                int(match.group(3)),
                int(match.group(4) or 1)
            )
        return None
    
    async def delete_file(self, file_path: str, force: bool = False) -> Dict[str, Any]:
        """
        Delete a file with backup.
        
        Args:
            file_path: Relative path to delete
            force: Force deletion without confirmation
            
        Returns:
            Result dictionary
        """
        full_path = self.workspace_path / file_path
        
        if not full_path.exists():
            return {
                'success': False,
                'error': f"File not found: {file_path}"
            }
        
        # Create backup before deletion
        backup_path = self._backup_file(full_path)
        
        # Delete file
        full_path.unlink()
        
        # Record in history
        version = FileVersion(
            path=file_path,
            content_hash='deleted',
            timestamp=datetime.now().isoformat(),
            size=0,
            action='delete'
        )
        self._history.append(version)
        self._save_history()
        
        self.logger.info(f"Deleted: {file_path} (backup: {backup_path})")
        
        return {
            'success': True,
            'path': file_path,
            'backup': backup_path
        }
    
    # ==================== Project Structure ====================
    
    async def get_project_structure(self) -> Dict[str, Any]:
        """
        Get current project structure.
        
        Returns:
            Nested dictionary of project structure
        """
        structure = {}
        
        for root, dirs, files in os.walk(self.workspace_path):
            # Skip hidden directories and common exclusions
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in 
                      ['__pycache__', 'node_modules', '.git', '.venv', 'venv']]
            
            rel_root = Path(root).relative_to(self.workspace_path)
            
            if str(rel_root) == '.':
                for f in files:
                    if not f.startswith('.'):
                        structure[f] = {'type': 'file'}
                for d in dirs:
                    structure[d] = {'type': 'directory', 'children': {}}
            else:
                # Navigate to correct position
                parts = rel_root.parts
                current = structure
                for part in parts:
                    if part in current and current[part]['type'] == 'directory':
                        if 'children' not in current[part]:
                            current[part]['children'] = {}
                        current = current[part]['children']
                
                for f in files:
                    if not f.startswith('.'):
                        current[f] = {'type': 'file'}
                for d in dirs:
                    current[d] = {'type': 'directory', 'children': {}}
        
        # Store in memory
        self.memory.set_project_structure(structure)
        
        return structure
    
    async def list_files(self, pattern: str = '*', directory: str = None) -> List[str]:
        """
        List files matching pattern.
        
        Args:
            pattern: Glob pattern
            directory: Subdirectory to search (optional)
            
        Returns:
            List of matching file paths
        """
        search_path = self.workspace_path
        if directory:
            search_path = search_path / directory
        
        files = list(search_path.glob(pattern))
        
        return [
            str(f.relative_to(self.workspace_path))
            for f in files
            if f.is_file() and not f.name.startswith('.')
        ]
    
    # ==================== Rollback ====================
    
    async def rollback_file(self, file_path: str, version_hash: str = None) -> Dict[str, Any]:
        """
        Rollback a file to a previous version.
        
        Args:
            file_path: File to rollback
            version_hash: Specific hash to rollback to (latest if None)
            
        Returns:
            Result dictionary
        """
        if version_hash:
            backup_name = f"{Path(file_path).name}.{version_hash}.bak"
            backup_path = self.backup_dir / backup_name
        else:
            # Find latest backup
            backups = list(self.backup_dir.glob(f"{Path(file_path).name}.*.bak"))
            if not backups:
                return {
                    'success': False,
                    'error': "No backup found"
                }
            backup_path = max(backups, key=lambda p: p.stat().st_mtime)
        
        if not backup_path.exists():
            return {
                'success': False,
                'error': f"Backup not found: {backup_path}"
            }
        
        full_path = self.workspace_path / file_path
        shutil.copy2(backup_path, full_path)
        
        self.logger.info(f"Rolled back: {file_path}")
        
        return {
            'success': True,
            'path': file_path,
            'restored_from': str(backup_path)
        }
    
    def get_file_history(self, file_path: str = None) -> List[FileVersion]:
        """Get history for a specific file or all files."""
        if file_path:
            return [v for v in self._history if v.path == file_path]
        return self._history.copy()


# Export
__all__ = ['FileManager', 'FileVersion']
