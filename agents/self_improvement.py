"""
Self Improvement Agent for continuous system improvement.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from agents.base_agent import BaseAgent, AgentTask


class SelfImprovementAgent(BaseAgent):
    """
    Self Improvement Agent responsible for:
    - Analyzing system performance
    - Identifying improvement opportunities
    - Auto-refactoring code
    - Learning from mistakes
    """
    
    def __init__(
        self,
        name: str = 'self_improvement',
        config: Optional[Dict[str, Any]] = None,
        model_manager = None,
        vector_memory = None
    ):
        super().__init__(name, config, model_manager, vector_memory)
        self._improvement_log: List[Dict] = []
        self._failed_attempts: List[Dict] = []
        self._successful_improvements: List[Dict] = []
    
    def _get_default_system_prompt(self) -> str:
        return """You are the Self-Improvement Agent in a quantitative trading system.
Your role is to:
1. Analyze system performance and identify weaknesses
2. Propose improvements to strategies and code
3. Learn from past mistakes and successes
4. Continuously optimize the system

Be thorough but conservative - only suggest changes with high confidence."""

    async def process_task(self, task: AgentTask) -> Any:
        """Process a self-improvement task."""
        if task.type == 'analyze_performance':
            return await self._analyze_performance(task)
        elif task.type == 'identify_improvements':
            return await self._identify_improvements(task)
        elif task.type == 'learn_from_failure':
            return await self._learn_from_failure(task)
        elif task.type == 'auto_refactor':
            return await self._auto_refactor(task)
        elif task.type == 'propose_changes':
            return await self._propose_changes(task)
        else:
            raise ValueError(f"Unknown task type: {task.type}")
    
    async def _analyze_performance(self, task: AgentTask) -> Dict[str, Any]:
        """Analyze system performance."""
        performance_data = task.parameters.get('performance_data', {})
        time_period = task.parameters.get('time_period', '30d')
        
        prompt = f"""Analyze this trading system performance:

Performance Data: {json.dumps(performance_data, indent=2, default=str)}
Time Period: {time_period}

Provide analysis including:
1. Overall performance assessment
2. Key strengths
3. Weaknesses and issues
4. Potential causes of underperformance
5. Prioritized improvement suggestions

Respond with JSON:
{{
    "overall_assessment": "good/poor/etc",
    "strengths": [],
    "weaknesses": [],
    "root_causes": [],
    "improvement_suggestions": [
        {{
            "priority": 1-5,
            "area": "strategy/execution/risk/etc",
            "suggestion": "specific suggestion",
            "expected_impact": "description"
        }}
    ]
}}"""

        response = await self.generate_response(prompt)
        
        try:
            json_str = response[response.find('{'):response.rfind('}')+1]
            analysis = json.loads(json_str)
        except:
            analysis = {
                'overall_assessment': 'unknown',
                'error': 'Could not parse analysis'
            }
        
        analysis['timestamp'] = datetime.now().isoformat()
        
        return analysis
    
    async def _identify_improvements(self, task: AgentTask) -> Dict[str, Any]:
        """Identify specific improvement opportunities."""
        context = task.parameters.get('context', {})
        focus_area = task.parameters.get('focus_area', 'all')
        
        # Get recent failures
        recent_failures = self._failed_attempts[-10:]
        
        # Get successful improvements
        successful = self._successful_improvements[-10:]
        
        prompt = f"""Identify improvement opportunities:

Context: {json.dumps(context, indent=2, default=str)}
Focus Area: {focus_area}

Recent Failures: {json.dumps(recent_failures, indent=2, default=str)}
Successful Improvements: {json.dumps(successful, indent=2, default=str)}

Provide specific, actionable improvements:
{{
    "improvements": [
        {{
            "id": "unique_id",
            "area": "strategy/code/risk/etc",
            "current_state": "description",
            "proposed_change": "specific change",
            "rationale": "why this will help",
            "risk": "potential risks",
            "confidence": 0-1
        }}
    ],
    "prioritization": ["ordered list of improvement ids"]
}}"""

        response = await self.generate_response(prompt)
        
        try:
            json_str = response[response.find('{'):response.rfind('}')+1]
            improvements = json.loads(json_str)
        except:
            improvements = {'improvements': [], 'error': 'Could not parse'}
        
        return improvements
    
    async def _learn_from_failure(self, task: AgentTask) -> Dict[str, Any]:
        """Learn from a failure event."""
        failure = task.parameters.get('failure', {})
        
        # Store failure
        failure_record = {
            **failure,
            'timestamp': datetime.now().isoformat()
        }
        self._failed_attempts.append(failure_record)
        
        prompt = f"""Analyze this failure and extract learnings:

Failure: {json.dumps(failure, indent=2, default=str)}

Provide:
1. Root cause analysis
2. What went wrong
3. How to prevent similar failures
4. Immediate actions needed
5. Long-term improvements

Respond with JSON:
{{
    "root_cause": "analysis",
    "what_went_wrong": [],
    "prevention_measures": [],
    "immediate_actions": [],
    "long_term_improvements": []
}}"""

        response = await self.generate_response(prompt)
        
        try:
            json_str = response[response.find('{'):response.rfind('}')+1]
            learnings = json.loads(json_str)
        except:
            learnings = {'root_cause': 'unknown', 'error': 'Could not analyze'}
        
        # Store in vector memory
        if self._vector_memory:
            self._vector_memory.store(
                content=f"Failure: {json.dumps(failure)}\nLearnings: {json.dumps(learnings)}",
                metadata={'type': 'failure_learning'}
            )
        
        return learnings
    
    async def _auto_refactor(self, task: AgentTask) -> Dict[str, Any]:
        """Automatically refactor code for improvement."""
        code = task.parameters.get('code', '')
        improvement_goal = task.parameters.get('goal', 'performance')
        
        prompt = f"""Refactor this code to improve {improvement_goal}:

```python
{code}
```

Provide:
1. Refactored code
2. Changes made
3. Expected improvements
4. Testing recommendations

Respond with JSON:
{{
    "refactored_code": "the improved code",
    "changes": ["list of changes"],
    "expected_improvements": [],
    "test_suggestions": []
}}"""

        response = await self.generate_response(prompt)
        
        try:
            json_str = response[response.find('{'):response.rfind('}')+1]
            refactoring = json.loads(json_str)
        except:
            refactoring = {'error': 'Could not parse refactoring'}
        
        return refactoring
    
    async def _propose_changes(self, task: AgentTask) -> Dict[str, Any]:
        """Propose changes based on analysis."""
        analysis_results = task.parameters.get('analysis_results', {})
        
        prompt = f"""Based on this analysis, propose specific changes:

{json.dumps(analysis_results, indent=2, default=str)}

For each proposed change, provide:
1. What to change
2. How to change it
3. Expected impact
4. Risk assessment
5. Rollback plan

Format as actionable proposals."""

        proposals = await self.generate_response(prompt)
        
        result = {
            'proposals': proposals,
            'timestamp': datetime.now().isoformat()
        }
        
        self._improvement_log.append(result)
        
        return result
    
    def record_success(self, improvement: Dict) -> None:
        """Record a successful improvement."""
        self._successful_improvements.append({
            **improvement,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_improvement_log(self, limit: int = 50) -> List[Dict]:
        """Get improvement log."""
        return self._improvement_log[-limit:]
    
    def get_failure_patterns(self) -> Dict[str, int]:
        """Analyze failure patterns."""
        patterns = {}
        
        for failure in self._failed_attempts:
            cause = failure.get('root_cause', 'unknown')
            patterns[cause] = patterns.get(cause, 0) + 1
        
        return patterns
    
    def get_status(self) -> Dict[str, Any]:
        """Get self-improvement agent status."""
        return {
            **super().get_status(),
            'improvements_proposed': len(self._improvement_log),
            'failures_analyzed': len(self._failed_attempts),
            'successful_improvements': len(self._successful_improvements)
        }
