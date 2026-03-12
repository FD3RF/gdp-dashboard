"""
Project State Manager
=====================

Manages the state of AI-driven software projects.
Tracks progress, decisions, and current context.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

from core.memory import MemoryStore


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


class ProjectPhase(Enum):
    PLANNING = "planning"
    ARCHITECTURE = "architecture"
    CODING = "coding"
    REVIEW = "review"
    TESTING = "testing"
    DEPLOYMENT = "deployment"


@dataclass
class Task:
    """Represents a development task."""
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    phase: ProjectPhase = ProjectPhase.PLANNING
    assigned_agent: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'description': self.description,
            'status': self.status.value,
            'phase': self.phase.value,
            'assigned_agent': self.assigned_agent,
            'dependencies': self.dependencies,
            'result': self.result,
            'error': self.error,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        return cls(
            id=data['id'],
            description=data['description'],
            status=TaskStatus(data['status']),
            phase=ProjectPhase(data['phase']),
            assigned_agent=data.get('assigned_agent'),
            dependencies=data.get('dependencies', []),
            result=data.get('result'),
            error=data.get('error'),
            created_at=data['created_at'],
            updated_at=data['updated_at']
        )


class ProjectStateManager:
    """
    Manages project state across the AI software factory.
    
    Features:
    - Task tracking with dependencies
    - Phase management
    - Progress tracking
    - Context persistence
    - Rollback support
    """
    
    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path)
        self.state_file = self.project_path / ".ai_factory" / "project_state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.memory = MemoryStore(str(self.state_file.parent))
        self.logger = logging.getLogger("ProjectState")
        
        # State containers
        self.tasks: Dict[str, Task] = {}
        self.phase = ProjectPhase.PLANNING
        self.context: Dict[str, Any] = {}
        self.decisions: List[Dict[str, Any]] = []
        self.artifacts: Dict[str, str] = {}  # name -> file path
        
        self._load()
    
    def _load(self):
        """Load state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                
                self.tasks = {
                    tid: Task.from_dict(t) 
                    for tid, t in data.get('tasks', {}).items()
                }
                self.phase = ProjectPhase(data.get('phase', 'planning'))
                self.context = data.get('context', {})
                self.decisions = data.get('decisions', [])
                self.artifacts = data.get('artifacts', {})
                
                self.logger.info(f"Loaded project state: {len(self.tasks)} tasks")
            except Exception as e:
                self.logger.warning(f"Failed to load state: {e}")
    
    def _save(self):
        """Save state to disk."""
        try:
            data = {
                'tasks': {tid: t.to_dict() for tid, t in self.tasks.items()},
                'phase': self.phase.value,
                'context': self.context,
                'decisions': self.decisions,
                'artifacts': self.artifacts,
                'updated_at': datetime.now().isoformat()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.debug("Project state saved")
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    # ==================== Task Management ====================
    
    def create_task(self, task_id: str, description: str, 
                   phase: ProjectPhase = None,
                   dependencies: List[str] = None) -> Task:
        """Create a new task."""
        task = Task(
            id=task_id,
            description=description,
            phase=phase or self.phase,
            dependencies=dependencies or []
        )
        self.tasks[task_id] = task
        self._save()
        
        self.logger.info(f"Created task: {task_id}")
        return task
    
    def start_task(self, task_id: str, agent: str = None) -> bool:
        """Mark task as started."""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        # Check dependencies
        for dep_id in task.dependencies:
            if dep_id in self.tasks:
                if self.tasks[dep_id].status != TaskStatus.COMPLETED:
                    self.logger.warning(f"Task {task_id} blocked by {dep_id}")
                    task.status = TaskStatus.BLOCKED
                    self._save()
                    return False
        
        task.status = TaskStatus.IN_PROGRESS
        task.assigned_agent = agent
        task.updated_at = datetime.now().isoformat()
        self._save()
        
        self.logger.info(f"Started task: {task_id} (agent: {agent})")
        return True
    
    def complete_task(self, task_id: str, result: str = None) -> bool:
        """Mark task as completed."""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        task.status = TaskStatus.COMPLETED
        task.result = result
        task.updated_at = datetime.now().isoformat()
        self._save()
        
        # Update memory
        self.memory.remember_decision(
            f"Completed task {task_id}",
            f"Result: {result}",
            "success"
        )
        
        self.logger.info(f"Completed task: {task_id}")
        return True
    
    def fail_task(self, task_id: str, error: str) -> bool:
        """Mark task as failed."""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        task.status = TaskStatus.FAILED
        task.error = error
        task.updated_at = datetime.now().isoformat()
        self._save()
        
        self.memory.remember_error(error, {'task': task_id})
        
        self.logger.error(f"Failed task: {task_id} - {error}")
        return True
    
    def get_next_task(self, phase: ProjectPhase = None) -> Optional[Task]:
        """Get next pending task for a phase."""
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING:
                if phase is None or task.phase == phase:
                    # Check dependencies
                    deps_met = all(
                        self.tasks.get(d, Task("", "")).status == TaskStatus.COMPLETED
                        for d in task.dependencies
                        if d in self.tasks
                    )
                    if deps_met:
                        return task
        return None
    
    def get_blocked_tasks(self) -> List[Task]:
        """Get all blocked tasks."""
        return [t for t in self.tasks.values() if t.status == TaskStatus.BLOCKED]
    
    # ==================== Phase Management ====================
    
    def set_phase(self, phase: ProjectPhase):
        """Set current project phase."""
        self.phase = phase
        self._save()
        self.logger.info(f"Phase changed to: {phase.value}")
    
    def advance_phase(self) -> bool:
        """Advance to next phase."""
        phases = list(ProjectPhase)
        current_idx = phases.index(self.phase)
        
        if current_idx < len(phases) - 1:
            self.phase = phases[current_idx + 1]
            self._save()
            self.logger.info(f"Advanced to phase: {self.phase.value}")
            return True
        return False
    
    # ==================== Context Management ====================
    
    def set_context(self, key: str, value: Any):
        """Set context value."""
        self.context[key] = value
        self._save()
    
    def get_context(self, key: str = None) -> Any:
        """Get context value(s)."""
        if key:
            return self.context.get(key)
        return self.context.copy()
    
    # ==================== Decision Tracking ====================
    
    def record_decision(self, decision: str, reasoning: str, 
                       alternatives: List[str] = None):
        """Record a project decision."""
        entry = {
            'decision': decision,
            'reasoning': reasoning,
            'alternatives': alternatives or [],
            'timestamp': datetime.now().isoformat()
        }
        self.decisions.append(entry)
        self._save()
        
        self.memory.remember_decision(decision, reasoning)
    
    def get_decisions(self) -> List[Dict[str, Any]]:
        """Get all recorded decisions."""
        return self.decisions.copy()
    
    # ==================== Artifact Tracking ====================
    
    def register_artifact(self, name: str, path: str):
        """Register a project artifact (file)."""
        self.artifacts[name] = path
        self._save()
    
    def get_artifact(self, name: str) -> Optional[str]:
        """Get artifact path."""
        return self.artifacts.get(name)
    
    # ==================== Progress ====================
    
    def get_progress(self) -> Dict[str, Any]:
        """Get project progress summary."""
        total = len(self.tasks)
        by_status = {}
        
        for status in TaskStatus:
            by_status[status.value] = sum(
                1 for t in self.tasks.values() if t.status == status
            )
        
        return {
            'phase': self.phase.value,
            'total_tasks': total,
            'by_status': by_status,
            'completion_rate': by_status.get('completed', 0) / total if total > 0 else 0,
            'blocked_count': by_status.get('blocked', 0),
            'failed_count': by_status.get('failed', 0)
        }
    
    def get_summary(self) -> str:
        """Get human-readable project summary."""
        progress = self.get_progress()
        
        summary = f"""
# Project Summary

**Phase:** {progress['phase']}
**Total Tasks:** {progress['total_tasks']}
**Completion:** {progress['completion_rate']*100:.1f}%

## Status Breakdown
"""
        for status, count in progress['by_status'].items():
            summary += f"- {status}: {count}\n"
        
        if progress['blocked_count'] > 0:
            summary += f"\n⚠️ {progress['blocked_count']} blocked tasks\n"
        
        if progress['failed_count'] > 0:
            summary += f"\n❌ {progress['failed_count']} failed tasks\n"
        
        # Add context summary
        context_summary = self.memory.get_context_summary()
        if context_summary:
            summary += f"\n## Context\n{context_summary}\n"
        
        return summary


# Export
__all__ = ['ProjectStateManager', 'Task', 'TaskStatus', 'ProjectPhase']
