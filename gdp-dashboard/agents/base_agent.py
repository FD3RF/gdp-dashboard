"""
Base Agent class for all AI agents.
"""

import asyncio
import logging
from abc import abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from core.base import BaseModule
from core.exceptions import AgentException, AgentTimeoutException


class AgentState(Enum):
    """Agent state enumeration."""
    IDLE = 'idle'
    THINKING = 'thinking'
    EXECUTING = 'executing'
    WAITING = 'waiting'
    ERROR = 'error'
    COMPLETED = 'completed'


@dataclass
class AgentTask:
    """Represents a task for an agent."""
    id: str
    type: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.type,
            'description': self.description,
            'parameters': self.parameters,
            'priority': self.priority,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error': self.error
        }


@dataclass
class AgentMemory:
    """Agent's working memory."""
    short_term: List[Dict[str, Any]] = field(default_factory=list)
    long_term: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def add_short_term(self, item: Dict[str, Any]) -> None:
        self.short_term.append(item)
        # Keep last 10 items
        if len(self.short_term) > 10:
            self.short_term = self.short_term[-10:]
    
    def add_long_term(self, item: Dict[str, Any]) -> None:
        self.long_term.append(item)
    
    def get_recent(self, n: int = 5) -> List[Dict[str, Any]]:
        return self.short_term[-n:]
    
    def clear_short_term(self) -> None:
        self.short_term.clear()


class BaseAgent(BaseModule):
    """
    Base class for all AI agents in the system.
    Provides common functionality for task execution, memory, and LLM integration.
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        model_manager = None,
        vector_memory = None
    ):
        super().__init__(name, config)
        self._state = AgentState.IDLE
        self._current_task: Optional[AgentTask] = None
        self._task_queue: List[AgentTask] = []
        self._memory = AgentMemory()
        self._model_manager = model_manager
        self._vector_memory = vector_memory
        self._max_retries = self.config.get('max_retries', 3)
        self._timeout = self.config.get('timeout', 300)
        self._callbacks: Dict[str, List[Callable]] = {
            'on_task_start': [],
            'on_task_complete': [],
            'on_task_error': [],
            'on_state_change': []
        }
    
    @property
    def state(self) -> AgentState:
        return self._state
    
    @state.setter
    def state(self, new_state: AgentState) -> None:
        old_state = self._state
        self._state = new_state
        self._notify_callbacks('on_state_change', old_state, new_state)
    
    def add_callback(self, event: str, callback: Callable) -> None:
        """Add an event callback."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def _notify_callbacks(self, event: str, *args, **kwargs) -> None:
        """Notify all callbacks for an event."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")
    
    async def initialize(self) -> bool:
        """Initialize the agent."""
        self.logger.info(f"Initializing agent: {self.name}")
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        """Start the agent."""
        self._running = True
        self._start_time = datetime.now()
        self.state = AgentState.IDLE
        return True
    
    async def stop(self) -> bool:
        """Stop the agent."""
        self._running = False
        self.state = AgentState.IDLE
        return True
    
    @abstractmethod
    async def process_task(self, task: AgentTask) -> Any:
        """
        Process a task. Must be implemented by subclasses.
        
        Args:
            task: Task to process
        
        Returns:
            Task result
        """
        pass
    
    async def submit_task(
        self,
        task_type: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
        priority: int = 1
    ) -> str:
        """
        Submit a new task to the agent.
        
        Args:
            task_type: Type of task
            description: Task description
            parameters: Task parameters
            priority: Task priority (1-10)
        
        Returns:
            Task ID
        """
        import uuid
        
        task = AgentTask(
            id=str(uuid.uuid4())[:8],
            type=task_type,
            description=description,
            parameters=parameters or {},
            priority=priority
        )
        
        self._task_queue.append(task)
        self._task_queue.sort(key=lambda t: t.priority, reverse=True)
        
        self.logger.debug(f"Submitted task: {task.id} - {task_type}")
        return task.id
    
    async def run_task(self, task_id: Optional[str] = None) -> Any:
        """
        Run a specific task or the next queued task.
        
        Args:
            task_id: Optional specific task ID
        
        Returns:
            Task result
        """
        if task_id:
            task = next((t for t in self._task_queue if t.id == task_id), None)
            if task:
                self._task_queue.remove(task)
        elif self._task_queue:
            task = self._task_queue.pop(0)
        else:
            raise AgentException("No tasks to execute", agent_name=self.name)
        
        return await self._execute_task(task)
    
    async def _execute_task(self, task: AgentTask) -> Any:
        """Execute a task with retry logic."""
        self._current_task = task
        task.started_at = datetime.now()
        self.state = AgentState.THINKING
        
        self._notify_callbacks('on_task_start', task)
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self.process_task(task),
                timeout=self._timeout
            )
            
            task.result = result
            task.completed_at = datetime.now()
            self.state = AgentState.COMPLETED
            
            # Store in memory
            self._memory.add_short_term({
                'task': task.to_dict(),
                'result': str(result)[:500]  # Truncate for storage
            })
            
            self._notify_callbacks('on_task_complete', task, result)
            
            return result
            
        except asyncio.TimeoutError:
            task.error = "Task timeout"
            self.state = AgentState.ERROR
            self._notify_callbacks('on_task_error', task, "timeout")
            raise AgentTimeoutException(self.name, self._timeout)
            
        except Exception as e:
            task.error = str(e)
            self.state = AgentState.ERROR
            self._notify_callbacks('on_task_error', task, str(e))
            raise AgentException(f"Task failed: {e}", agent_name=self.name)
            
        finally:
            self._current_task = None
            self.state = AgentState.IDLE
    
    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate response using the LLM.
        
        Args:
            prompt: Input prompt
            system_prompt: System prompt
            **kwargs: Additional generation parameters
        
        Returns:
            Generated response
        """
        if not self._model_manager:
            raise AgentException("Model manager not configured", agent_name=self.name)
        
        self.state = AgentState.THINKING
        
        try:
            response = await self._model_manager.generate(
                prompt=prompt,
                system_prompt=system_prompt or self._get_default_system_prompt(),
                **kwargs
            )
            
            return response.content
            
        except Exception as e:
            raise AgentException(f"Generation failed: {e}", agent_name=self.name)
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for the agent."""
        return f"You are {self.name}, an AI agent in a quantitative trading system."
    
    def add_to_memory(self, key: str, value: Any) -> None:
        """Add item to agent memory."""
        self._memory.context[key] = value
    
    def get_from_memory(self, key: str) -> Optional[Any]:
        """Get item from agent memory."""
        return self._memory.context.get(key)
    
    async def store_long_term_memory(
        self,
        content: str,
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """Store item in long-term vector memory."""
        if self._vector_memory:
            return self._vector_memory.store(
                content=content,
                metadata={'agent': self.name, **(metadata or {})}
            )
        return None
    
    async def recall_from_memory(
        self,
        query: str,
        k: int = 5
    ) -> List[Any]:
        """Recall from long-term memory."""
        if self._vector_memory:
            results = self._vector_memory.retrieve(query, k=k)
            return [r[0] for r in results]
        return []
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            **super().get_status(),
            'state': self.state.value,
            'current_task': self._current_task.to_dict() if self._current_task else None,
            'queued_tasks': len(self._task_queue),
            'memory_items': len(self._memory.short_term)
        }
    
    def clear_queue(self) -> None:
        """Clear task queue."""
        self._task_queue.clear()
    
    def get_queued_tasks(self) -> List[Dict]:
        """Get all queued tasks."""
        return [t.to_dict() for t in self._task_queue]
