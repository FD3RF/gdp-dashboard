"""
Task Queue module for async task processing.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple
import heapq
import json
from core.base import BaseModule
from core.exceptions import QuantSystemException


class TaskPriority(IntEnum):
    """Task priority levels (lower = higher priority)."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class TaskStatus(IntEnum):
    """Task status enumeration."""
    PENDING = 0
    QUEUED = 1
    RUNNING = 2
    COMPLETED = 3
    FAILED = 4
    CANCELLED = 5
    RETRY = 6


@dataclass(order=True)
class Task:
    """Represents a task in the queue."""
    priority: int
    created_at: float = field(compare=True)
    task_id: str = field(compare=False)
    name: str = field(compare=False)
    func: Callable = field(compare=False)
    args: Tuple = field(compare=False, default=())
    kwargs: Dict = field(compare=False, default_factory=dict)
    status: TaskStatus = field(compare=False, default=TaskStatus.PENDING)
    result: Any = field(compare=False, default=None)
    error: Optional[str] = field(compare=False, default=None)
    retries: int = field(compare=False, default=0)
    max_retries: int = field(compare=False, default=3)
    timeout: int = field(compare=False, default=300)
    started_at: Optional[float] = field(compare=False, default=None)
    completed_at: Optional[float] = field(compare=False, default=None)
    metadata: Dict = field(compare=False, default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            'task_id': self.task_id,
            'name': self.name,
            'priority': self.priority,
            'status': self.status.name,
            'created_at': datetime.fromtimestamp(self.created_at).isoformat(),
            'started_at': datetime.fromtimestamp(self.started_at).isoformat() if self.started_at else None,
            'completed_at': datetime.fromtimestamp(self.completed_at).isoformat() if self.completed_at else None,
            'retries': self.retries,
            'error': self.error,
            'metadata': self.metadata
        }


class TaskQueue(BaseModule):
    """
    Priority-based async task queue with worker pool.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('task_queue', config)
        self._queue: List[Task] = []
        self._tasks: Dict[str, Task] = {}
        self._workers: List[asyncio.Task] = []
        self._worker_count = self.config.get('worker_count', 4)
        self._max_queue_size = self.config.get('max_queue_size', 1000)
        self._running = False
        self._queue_lock = asyncio.Lock()
        self._task_available = asyncio.Event()
        self._completed_tasks: Dict[str, Task] = {}
        self._max_completed_tasks = self.config.get('max_completed_tasks', 1000)
    
    async def initialize(self) -> bool:
        """Initialize the task queue."""
        self.logger.info(f"Initializing task queue with {self._worker_count} workers...")
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        """Start the task queue workers."""
        if self._running:
            return True
        
        self.logger.info("Starting task queue workers...")
        self._running = True
        self._start_time = datetime.now()
        
        # Start worker coroutines
        for i in range(self._worker_count):
            worker = asyncio.create_task(self._worker(i))
            self._workers.append(worker)
        
        return True
    
    async def stop(self) -> bool:
        """Stop the task queue."""
        self.logger.info("Stopping task queue...")
        self._running = False
        
        # Signal workers to stop
        self._task_available.set()
        
        # Wait for workers to finish
        for worker in self._workers:
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
        
        self._workers.clear()
        return True
    
    async def _worker(self, worker_id: int) -> None:
        """Worker coroutine that processes tasks."""
        self.logger.debug(f"Worker {worker_id} started")
        
        while self._running:
            try:
                # Wait for task to be available
                await self._task_available.wait()
                
                # Get next task
                async with self._queue_lock:
                    if not self._queue:
                        self._task_available.clear()
                        continue
                    
                    task = heapq.heappop(self._queue)
                    task.status = TaskStatus.RUNNING
                    task.started_at = time.time()
                
                self.logger.debug(f"Worker {worker_id} processing task: {task.name}")
                
                # Execute task
                try:
                    result = await asyncio.wait_for(
                        self._execute_task(task),
                        timeout=task.timeout
                    )
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = time.time()
                except asyncio.TimeoutError:
                    task.error = "Task timeout"
                    task.status = TaskStatus.FAILED
                    self.logger.error(f"Task {task.name} timed out")
                except Exception as e:
                    task.error = str(e)
                    if task.retries < task.max_retries:
                        task.status = TaskStatus.RETRY
                        task.retries += 1
                        # Re-queue with delay
                        await asyncio.sleep(2 ** task.retries)
                        await self._enqueue(task)
                    else:
                        task.status = TaskStatus.FAILED
                        self.logger.error(f"Task {task.name} failed: {e}")
                
                # Store completed task
                self._completed_tasks[task.task_id] = task
                if len(self._completed_tasks) > self._max_completed_tasks:
                    # Remove oldest completed task
                    oldest = next(iter(self._completed_tasks))
                    del self._completed_tasks[oldest]
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
        
        self.logger.debug(f"Worker {worker_id} stopped")
    
    async def _execute_task(self, task: Task) -> Any:
        """Execute a task function."""
        if asyncio.iscoroutinefunction(task.func):
            return await task.func(*task.args, **task.kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: task.func(*task.args, **task.kwargs)
            )
    
    async def _enqueue(self, task: Task) -> bool:
        """Add task to the queue."""
        async with self._queue_lock:
            if len(self._queue) >= self._max_queue_size:
                raise QuantSystemException("Task queue is full")
            
            heapq.heappush(self._queue, task)
            task.status = TaskStatus.QUEUED
            self._tasks[task.task_id] = task
            self._task_available.set()
        
        return True
    
    def submit(
        self,
        name: str,
        func: Callable,
        args: Tuple = (),
        kwargs: Optional[Dict] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: int = 300,
        max_retries: int = 3,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Submit a new task to the queue.
        
        Args:
            name: Task name
            func: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            priority: Task priority
            timeout: Timeout in seconds
            max_retries: Maximum retry attempts
            metadata: Additional metadata
        
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())[:12]
        
        task = Task(
            priority=priority.value,
            created_at=time.time(),
            task_id=task_id,
            name=name,
            func=func,
            args=args,
            kwargs=kwargs or {},
            timeout=timeout,
            max_retries=max_retries,
            metadata=metadata or {}
        )
        
        # Use asyncio to add to queue
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(self._enqueue(task))
        else:
            loop.run_until_complete(self._enqueue(task))
        
        self.logger.debug(f"Submitted task: {name} (ID: {task_id})")
        return task_id
    
    async def submit_async(
        self,
        name: str,
        func: Callable,
        args: Tuple = (),
        kwargs: Optional[Dict] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: int = 300,
        max_retries: int = 3,
        metadata: Optional[Dict] = None
    ) -> str:
        """Submit a task asynchronously."""
        task_id = str(uuid.uuid4())[:12]
        
        task = Task(
            priority=priority.value,
            created_at=time.time(),
            task_id=task_id,
            name=name,
            func=func,
            args=args,
            kwargs=kwargs or {},
            timeout=timeout,
            max_retries=max_retries,
            metadata=metadata or {}
        )
        
        await self._enqueue(task)
        return task_id
    
    async def get_result(self, task_id: str, timeout: float = None) -> Any:
        """
        Get task result, waiting if necessary.
        
        Args:
            task_id: Task ID
            timeout: Maximum wait time
        
        Returns:
            Task result
        
        Raises:
            QuantSystemException: If task failed or not found
        """
        start_time = time.time()
        
        while True:
            # Check completed tasks
            if task_id in self._completed_tasks:
                task = self._completed_tasks[task_id]
                if task.status == TaskStatus.COMPLETED:
                    return task.result
                elif task.status == TaskStatus.FAILED:
                    raise QuantSystemException(f"Task failed: {task.error}")
                elif task.status == TaskStatus.CANCELLED:
                    raise QuantSystemException("Task was cancelled")
            
            # Check pending tasks
            if task_id in self._tasks:
                task = self._tasks[task_id]
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    continue
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise QuantSystemException("Timeout waiting for task result")
            
            await asyncio.sleep(0.1)
    
    def get_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status."""
        if task_id in self._completed_tasks:
            return self._completed_tasks[task_id].to_dict()
        if task_id in self._tasks:
            return self._tasks[task_id].to_dict()
        return None
    
    def cancel(self, task_id: str) -> bool:
        """Cancel a pending task."""
        if task_id in self._tasks:
            task = self._tasks[task_id]
            if task.status in (TaskStatus.PENDING, TaskStatus.QUEUED):
                task.status = TaskStatus.CANCELLED
                return True
        return False
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return len(self._queue)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        pending = sum(1 for t in self._tasks.values() if t.status in (TaskStatus.PENDING, TaskStatus.QUEUED))
        running = sum(1 for t in self._tasks.values() if t.status == TaskStatus.RUNNING)
        completed = sum(1 for t in self._completed_tasks.values() if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self._completed_tasks.values() if t.status == TaskStatus.FAILED)
        
        return {
            'queue_size': len(self._queue),
            'pending': pending,
            'running': running,
            'completed': completed,
            'failed': failed,
            'workers': self._worker_count
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        stats = self.get_stats()
        return {
            'healthy': self._running and len(self._workers) == self._worker_count,
            'name': self.name,
            'timestamp': datetime.now().isoformat(),
            'stats': stats
        }
