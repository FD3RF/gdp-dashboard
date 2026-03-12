"""
Memory Agent for managing agent memories and knowledge.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from agents.base_agent import BaseAgent, AgentTask


class MemoryAgent(BaseAgent):
    """
    Memory Agent responsible for:
    - Managing agent memories
    - Knowledge retrieval
    - Experience storage
    """
    
    def __init__(
        self,
        name: str = 'memory',
        config: Optional[Dict[str, Any]] = None,
        model_manager = None,
        vector_memory = None
    ):
        super().__init__(name, config, model_manager, vector_memory)
        self._experiences: List[Dict] = []
        self._knowledge_base: Dict[str, Dict] = {}
    
    def _get_default_system_prompt(self) -> str:
        return """You are the Memory Agent in a quantitative trading system.
Your role is to:
1. Store and retrieve agent experiences
2. Manage the knowledge base
3. Identify patterns in historical data
4. Provide context for other agents

Help other agents learn from past experiences."""

    async def process_task(self, task: AgentTask) -> Any:
        """Process a memory task."""
        if task.type == 'store_experience':
            return await self._store_experience(task)
        elif task.type == 'retrieve_similar':
            return await self._retrieve_similar(task)
        elif task.type == 'summarize':
            return await self._summarize(task)
        elif task.type == 'add_knowledge':
            return await self._add_knowledge(task)
        elif task.type == 'query_knowledge':
            return await self._query_knowledge(task)
        else:
            raise ValueError(f"Unknown task type: {task.type}")
    
    async def _store_experience(self, task: AgentTask) -> Dict[str, Any]:
        """Store an experience in memory."""
        agent_name = task.parameters.get('agent_name', 'unknown')
        experience_type = task.parameters.get('type', 'general')
        content = task.parameters.get('content', {})
        outcome = task.parameters.get('outcome', {})
        
        experience = {
            'id': f"exp_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'agent_name': agent_name,
            'type': experience_type,
            'content': content,
            'outcome': outcome,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in local experiences
        self._experiences.append(experience)
        
        # Store in vector memory if available
        if self._vector_memory:
            text = json.dumps(content) + " " + json.dumps(outcome)
            memory_id = self._vector_memory.store(
                content=text,
                metadata={
                    'agent': agent_name,
                    'type': experience_type,
                    'experience_id': experience['id']
                }
            )
            experience['vector_memory_id'] = memory_id
        
        return experience
    
    async def _retrieve_similar(self, task: AgentTask) -> List[Dict[str, Any]]:
        """Retrieve similar experiences."""
        query = task.parameters.get('query', '')
        k = task.parameters.get('k', 5)
        agent_filter = task.parameters.get('agent_filter')
        
        if self._vector_memory:
            # Use vector search
            results = self._vector_memory.retrieve(query, k=k)
            
            experiences = []
            for memory, score in results:
                exp = {
                    'content': memory.content,
                    'score': score,
                    'metadata': memory.metadata
                }
                
                if agent_filter and memory.metadata.get('agent') != agent_filter:
                    continue
                    
                experiences.append(exp)
            
            return experiences
        
        # Fallback to keyword search
        query_lower = query.lower()
        matching = []
        
        for exp in self._experiences:
            content_str = json.dumps(exp['content']).lower()
            if query_lower in content_str:
                if agent_filter and exp['agent_name'] != agent_filter:
                    continue
                matching.append(exp)
        
        return matching[:k]
    
    async def _summarize(self, task: AgentTask) -> Dict[str, Any]:
        """Summarize experiences for a time period."""
        agent_name = task.parameters.get('agent_name')
        days = task.parameters.get('days', 7)
        
        # Filter experiences
        cutoff = datetime.now() - timedelta(days=days)
        relevant = [
            exp for exp in self._experiences
            if datetime.fromisoformat(exp['timestamp']) > cutoff
        ]
        
        if agent_name:
            relevant = [exp for exp in relevant if exp['agent_name'] == agent_name]
        
        if not relevant:
            return {'summary': 'No experiences in the specified period'}
        
        # Generate summary using LLM
        prompt = f"""Summarize these trading system experiences:

{json.dumps(relevant[-20:], indent=2, default=str)}

Provide:
1. Key learnings
2. Common patterns
3. Areas for improvement
4. Success factors"""

        summary = await self.generate_response(prompt)
        
        return {
            'period_days': days,
            'experience_count': len(relevant),
            'agent_name': agent_name,
            'summary': summary
        }
    
    async def _add_knowledge(self, task: AgentTask) -> Dict[str, Any]:
        """Add knowledge to the knowledge base."""
        topic = task.parameters.get('topic')
        content = task.parameters.get('content')
        source = task.parameters.get('source', 'manual')
        
        knowledge = {
            'topic': topic,
            'content': content,
            'source': source,
            'created_at': datetime.now().isoformat()
        }
        
        self._knowledge_base[topic] = knowledge
        
        # Store in vector memory
        if self._vector_memory:
            self._vector_memory.store(
                content=f"{topic}: {content}",
                metadata={'type': 'knowledge', 'source': source}
            )
        
        return knowledge
    
    async def _query_knowledge(self, task: AgentTask) -> Dict[str, Any]:
        """Query the knowledge base."""
        query = task.parameters.get('query', '')
        
        # Check direct topic match
        for topic, knowledge in self._knowledge_base.items():
            if query.lower() in topic.lower():
                return knowledge
        
        # Use vector search
        if self._vector_memory:
            results = self._vector_memory.retrieve(query, k=3)
            if results:
                return {
                    'query': query,
                    'results': [
                        {
                            'content': r[0].content,
                            'score': r[1],
                            'metadata': r[0].metadata
                        }
                        for r in results
                    ]
                }
        
        return {'query': query, 'results': []}
    
    def get_experiences(self, agent_name: Optional[str] = None) -> List[Dict]:
        """Get all experiences."""
        if agent_name:
            return [e for e in self._experiences if e['agent_name'] == agent_name]
        return self._experiences
    
    def get_knowledge_topics(self) -> List[str]:
        """Get all knowledge topics."""
        return list(self._knowledge_base.keys())
    
    def get_status(self) -> Dict[str, Any]:
        """Get memory agent status."""
        return {
            **super().get_status(),
            'experience_count': len(self._experiences),
            'knowledge_topics': len(self._knowledge_base),
            'has_vector_memory': self._vector_memory is not None
        }


from datetime import timedelta
