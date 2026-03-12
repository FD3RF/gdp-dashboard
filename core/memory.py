"""
Memory System for AI Agents
============================

Long-term memory for maintaining context across agent interactions.
Prevents AI from forgetting project state, repeating tasks, or losing context.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import logging


class MemoryType(Enum):
    """Types of memory entries."""
    PROJECT_CONTEXT = "project_context"
    TASK_STATE = "task_state"
    CODE_SNIPPET = "code_snippet"
    ERROR_LOG = "error_log"
    DECISION = "decision"
    LEARNING = "learning"


@dataclass
class MemoryEntry:
    """A single memory entry."""
    id: str
    type: MemoryType
    content: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 1.0  # 0.0 - 1.0, higher = more important
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.type.value,
            'content': self.content,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
            'importance': self.importance
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        return cls(
            id=data['id'],
            type=MemoryType(data['type']),
            content=data['content'],
            timestamp=data['timestamp'],
            metadata=data.get('metadata', {}),
            importance=data.get('importance', 1.0)
        )


class MemoryStore:
    """
    Long-term memory store for AI agents.
    
    Features:
    - Project context persistence
    - Task state tracking
    - Error history
    - Decision logging
    - Learning accumulation
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, storage_path: Optional[str] = None):
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.storage_path = Path(storage_path or ".ai_memory")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.project_context_file = self.storage_path / "project_context.json"
        self.task_state_file = self.storage_path / "task_state.json"
        self.memory_file = self.storage_path / "memory_store.json"
        
        self._memories: List[MemoryEntry] = []
        self._project_context: Dict[str, Any] = {}
        self._task_states: Dict[str, Any] = {}
        
        self.logger = logging.getLogger("MemoryStore")
        self._load()
        self._initialized = True
    
    def _load(self):
        """Load memory from disk."""
        try:
            if self.project_context_file.exists():
                with open(self.project_context_file, 'r') as f:
                    self._project_context = json.load(f)
            
            if self.task_state_file.exists():
                with open(self.task_state_file, 'r') as f:
                    self._task_states = json.load(f)
            
            if self.memory_file.exists():
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self._memories = [MemoryEntry.from_dict(m) for m in data]
                    
            self.logger.info(f"Loaded {len(self._memories)} memories from disk")
        except Exception as e:
            self.logger.warning(f"Failed to load memory: {e}")
    
    def _save(self):
        """Save memory to disk."""
        try:
            with open(self.project_context_file, 'w') as f:
                json.dump(self._project_context, f, indent=2)
            
            with open(self.task_state_file, 'w') as f:
                json.dump(self._task_states, f, indent=2)
            
            with open(self.memory_file, 'w') as f:
                json.dump([m.to_dict() for m in self._memories], f, indent=2)
                
            self.logger.debug("Memory saved to disk")
        except Exception as e:
            self.logger.error(f"Failed to save memory: {e}")
    
    # ==================== Project Context ====================
    
    def update_project_context(self, key: str, value: Any):
        """Update project context."""
        self._project_context[key] = value
        self._project_context['last_updated'] = datetime.now().isoformat()
        self._save()
    
    def get_project_context(self, key: str = None) -> Any:
        """Get project context."""
        if key:
            return self._project_context.get(key)
        return self._project_context
    
    def set_project_structure(self, structure: Dict[str, Any]):
        """Record project structure to avoid AI forgetting modules."""
        self._project_context['structure'] = structure
        self._save()
    
    def get_project_structure(self) -> Dict[str, Any]:
        """Get recorded project structure."""
        return self._project_context.get('structure', {})
    
    # ==================== Task State ====================
    
    def save_task_state(self, task_id: str, state: Dict[str, Any]):
        """Save current task state."""
        state['updated_at'] = datetime.now().isoformat()
        self._task_states[task_id] = state
        self._save()
    
    def get_task_state(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task state by ID."""
        return self._task_states.get(task_id)
    
    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """Get all active (incomplete) tasks."""
        return [
            {'id': tid, **state}
            for tid, state in self._task_states.items()
            if state.get('status') != 'completed'
        ]
    
    def clear_task_state(self, task_id: str):
        """Clear completed task state."""
        if task_id in self._task_states:
            self._task_states[task_id]['status'] = 'completed'
            self._save()
    
    # ==================== Memory Operations ====================
    
    def remember(self, memory_type: MemoryType, content: Dict[str, Any], 
                 metadata: Dict[str, Any] = None, importance: float = 1.0) -> str:
        """
        Store a new memory entry.
        
        Args:
            memory_type: Type of memory
            content: Memory content
            metadata: Additional metadata
            importance: Importance score (0.0-1.0)
            
        Returns:
            Memory ID
        """
        # Generate unique ID
        content_hash = hashlib.md5(
            json.dumps(content, sort_keys=True).encode()
        ).hexdigest()[:8]
        memory_id = f"{memory_type.value}_{content_hash}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        entry = MemoryEntry(
            id=memory_id,
            type=memory_type,
            content=content,
            metadata=metadata or {},
            importance=importance
        )
        
        self._memories.append(entry)
        self._save()
        
        self.logger.info(f"Stored memory: {memory_id}")
        return memory_id
    
    def recall(self, memory_type: MemoryType = None, limit: int = 10) -> List[MemoryEntry]:
        """
        Recall memories, optionally filtered by type.
        
        Args:
            memory_type: Filter by type (optional)
            limit: Maximum number of memories to return
            
        Returns:
            List of memory entries, sorted by importance and recency
        """
        memories = self._memories
        
        if memory_type:
            memories = [m for m in memories if m.type == memory_type]
        
        # Sort by importance (descending) then by timestamp (descending)
        memories = sorted(
            memories,
            key=lambda m: (m.importance, m.timestamp),
            reverse=True
        )
        
        return memories[:limit]
    
    def recall_errors(self, limit: int = 20) -> List[MemoryEntry]:
        """Recall error history to avoid repeating mistakes."""
        return self.recall(MemoryType.ERROR_LOG, limit)
    
    def recall_decisions(self, limit: int = 20) -> List[MemoryEntry]:
        """Recall past decisions."""
        return self.recall(MemoryType.DECISION, limit)
    
    def remember_error(self, error: str, context: Dict[str, Any] = None):
        """Remember an error for future reference."""
        self.remember(
            MemoryType.ERROR_LOG,
            {'error': error, 'context': context or {}},
            importance=0.8
        )
    
    def remember_decision(self, decision: str, reasoning: str, outcome: str = None):
        """Remember a decision and its reasoning."""
        self.remember(
            MemoryType.DECISION,
            {
                'decision': decision,
                'reasoning': reasoning,
                'outcome': outcome
            },
            importance=0.9
        )
    
    def remember_learning(self, lesson: str, context: str):
        """Remember a learned lesson."""
        self.remember(
            MemoryType.LEARNING,
            {'lesson': lesson, 'context': context},
            importance=1.0
        )
    
    # ==================== Context Summary ====================
    
    def get_context_summary(self) -> str:
        """Get a summary of current context for AI prompts."""
        summary_parts = []
        
        # Project context
        if self._project_context:
            summary_parts.append("=== PROJECT CONTEXT ===")
            for key, value in self._project_context.items():
                if key != 'structure':
                    summary_parts.append(f"{key}: {value}")
        
        # Active tasks
        active_tasks = self.get_active_tasks()
        if active_tasks:
            summary_parts.append("\n=== ACTIVE TASKS ===")
            for task in active_tasks[:5]:
                summary_parts.append(f"- {task.get('id')}: {task.get('description', 'N/A')}")
        
        # Recent learnings
        learnings = self.recall(MemoryType.LEARNING, limit=3)
        if learnings:
            summary_parts.append("\n=== RECENT LEARNINGS ===")
            for learning in learnings:
                summary_parts.append(f"- {learning.content.get('lesson')}")
        
        # Recent errors to avoid
        errors = self.recall_errors(limit=3)
        if errors:
            summary_parts.append("\n=== ERRORS TO AVOID ===")
            for error in errors:
                summary_parts.append(f"- {error.content.get('error')[:100]}")
        
        return "\n".join(summary_parts) if summary_parts else "No context available."
    
    def clear_old_memories(self, days: int = 30):
        """Clear memories older than specified days."""
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        original_count = len(self._memories)
        self._memories = [
            m for m in self._memories
            if datetime.fromisoformat(m.timestamp).timestamp() > cutoff or m.importance > 0.8
        ]
        
        if len(self._memories) < original_count:
            self._save()
            self.logger.info(f"Cleared {original_count - len(self._memories)} old memories")


# Global memory store instance
memory_store = MemoryStore()
