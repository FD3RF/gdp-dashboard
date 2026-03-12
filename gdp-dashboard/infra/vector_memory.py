"""
Vector Memory module for AI agent memory storage.
"""

import asyncio
import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from core.base import BaseModule
from core.exceptions import AgentException


@dataclass
class MemoryEntry:
    """Represents a memory entry."""
    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    importance: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'content': self.content,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'importance': self.importance
        }


class VectorMemory(BaseModule):
    """
    Vector-based memory system for AI agents.
    Supports semantic search, importance ranking, and memory decay.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('vector_memory', config)
        self._memories: Dict[str, MemoryEntry] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        self._embedding_dim = self.config.get('embedding_dim', 768)
        self._max_memories = self.config.get('max_memories', 10000)
        self._similarity_threshold = self.config.get('similarity_threshold', 0.7)
        self._decay_rate = self.config.get('decay_rate', 0.01)
        self._embedding_model = None
    
    async def initialize(self) -> bool:
        """Initialize the vector memory."""
        self.logger.info("Initializing vector memory...")
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        """Start the vector memory."""
        self._running = True
        self._start_time = datetime.now()
        return True
    
    async def stop(self) -> bool:
        """Stop the vector memory."""
        self._running = False
        return True
    
    def _compute_embedding(self, text: str) -> np.ndarray:
        """
        Compute embedding for text.
        In production, this would use a real embedding model.
        For now, uses a simple hash-based embedding.
        """
        # Simple hash-based embedding (replace with actual embedding model)
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self._embedding_dim)
        return embedding / np.linalg.norm(embedding)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def store(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
        embedding: Optional[np.ndarray] = None
    ) -> str:
        """
        Store a new memory.
        
        Args:
            content: Memory content
            metadata: Optional metadata
            importance: Importance score (0-1)
            embedding: Optional pre-computed embedding
        
        Returns:
            Memory ID
        """
        import uuid
        memory_id = str(uuid.uuid4())[:12]
        
        # Compute embedding if not provided
        if embedding is None:
            embedding = self._compute_embedding(content)
        
        entry = MemoryEntry(
            id=memory_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            importance=importance
        )
        
        self._memories[memory_id] = entry
        self._embeddings[memory_id] = embedding
        
        # Enforce max memories
        if len(self._memories) > self._max_memories:
            self._evict_memories()
        
        self.logger.debug(f"Stored memory: {memory_id}")
        return memory_id
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        threshold: Optional[float] = None
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Retrieve memories similar to query.
        
        Args:
            query: Query text
            k: Number of results
            threshold: Similarity threshold
        
        Returns:
            List of (memory, similarity) tuples
        """
        if not self._memories:
            return []
        
        threshold = threshold or self._similarity_threshold
        query_embedding = self._compute_embedding(query)
        
        # Compute similarities
        similarities = []
        for memory_id, memory in self._memories.items():
            if memory.embedding is not None:
                sim = self._cosine_similarity(query_embedding, memory.embedding)
                if sim >= threshold:
                    similarities.append((memory, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Update access counts
        for memory, _ in similarities[:k]:
            memory.access_count += 1
            memory.last_accessed = datetime.now()
        
        return similarities[:k]
    
    def retrieve_by_metadata(
        self,
        metadata_filter: Dict[str, Any],
        k: int = 10
    ) -> List[MemoryEntry]:
        """
        Retrieve memories matching metadata filter.
        
        Args:
            metadata_filter: Metadata key-value pairs to match
            k: Maximum results
        
        Returns:
            List of matching memories
        """
        results = []
        for memory in self._memories.values():
            match = True
            for key, value in metadata_filter.items():
                if memory.metadata.get(key) != value:
                    match = False
                    break
            if match:
                results.append(memory)
        
        # Sort by importance and recency
        results.sort(key=lambda m: (m.importance, m.created_at), reverse=True)
        return results[:k]
    
    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory by ID."""
        memory = self._memories.get(memory_id)
        if memory:
            memory.access_count += 1
            memory.last_accessed = datetime.now()
        return memory
    
    def update(
        self,
        memory_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: Optional[float] = None
    ) -> bool:
        """Update an existing memory."""
        if memory_id not in self._memories:
            return False
        
        memory = self._memories[memory_id]
        
        if content is not None:
            memory.content = content
            memory.embedding = self._compute_embedding(content)
            self._embeddings[memory_id] = memory.embedding
        
        if metadata is not None:
            memory.metadata.update(metadata)
        
        if importance is not None:
            memory.importance = importance
        
        return True
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        if memory_id in self._memories:
            del self._memories[memory_id]
            if memory_id in self._embeddings:
                del self._embeddings[memory_id]
            return True
        return False
    
    def _evict_memories(self) -> int:
        """
        Evict least important memories.
        
        Returns:
            Number of evicted memories
        """
        # Score memories by importance, access count, and recency
        scores = []
        for memory_id, memory in self._memories.items():
            age_hours = (datetime.now() - memory.created_at).total_seconds() / 3600
            recency_score = 1 / (1 + age_hours * self._decay_rate)
            
            score = (
                memory.importance * 0.4 +
                (memory.access_count / 100) * 0.3 +
                recency_score * 0.3
            )
            scores.append((memory_id, score))
        
        # Sort by score and remove lowest
        scores.sort(key=lambda x: x[1])
        to_remove = len(self._memories) - self._max_memories + 100
        
        for memory_id, _ in scores[:to_remove]:
            self.delete(memory_id)
        
        self.logger.debug(f"Evicted {to_remove} memories")
        return to_remove
    
    def clear(self) -> None:
        """Clear all memories."""
        self._memories.clear()
        self._embeddings.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        total_importance = sum(m.importance for m in self._memories.values())
        avg_importance = total_importance / len(self._memories) if self._memories else 0
        
        return {
            'total_memories': len(self._memories),
            'max_memories': self._max_memories,
            'embedding_dim': self._embedding_dim,
            'avg_importance': avg_importance,
            'similarity_threshold': self._similarity_threshold
        }
    
    def export_memories(self) -> List[Dict[str, Any]]:
        """Export all memories as list of dicts."""
        return [m.to_dict() for m in self._memories.values()]
    
    def import_memories(self, memories: List[Dict[str, Any]]) -> int:
        """Import memories from list of dicts."""
        count = 0
        for mem_dict in memories:
            try:
                self.store(
                    content=mem_dict['content'],
                    metadata=mem_dict.get('metadata', {}),
                    importance=mem_dict.get('importance', 0.5)
                )
                count += 1
            except Exception as e:
                self.logger.error(f"Error importing memory: {e}")
        return count
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            'healthy': self._initialized,
            'name': self.name,
            'timestamp': datetime.now().isoformat(),
            'stats': self.get_stats()
        }
