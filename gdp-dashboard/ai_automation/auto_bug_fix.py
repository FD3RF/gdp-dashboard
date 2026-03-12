"""
Auto Bug Fix Agent for automated code debugging.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from core.base import BaseModule


class AutoBugFixAgent(BaseModule):
    """
    Automatically detects and fixes code bugs.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('auto_bug_fix_agent', config)
        
        self._model_manager = None
        self._bug_history: List[Dict] = []
    
    def set_model_manager(self, manager) -> None:
        """Set AI model manager."""
        self._model_manager = manager
    
    async def initialize(self) -> bool:
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        self._running = True
        return True
    
    async def stop(self) -> bool:
        self._running = False
        return True
    
    async def analyze_error(
        self,
        code: str,
        error_message: str,
        stack_trace: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze an error and provide diagnosis.
        
        Args:
            code: Source code with error
            error_message: Error message
            stack_trace: Optional stack trace
        
        Returns:
            Error diagnosis
        """
        prompt = f"""Analyze this Python error and provide a diagnosis:

Code:
```python
{code}
```

Error: {error_message}
Stack Trace: {stack_trace or 'Not provided'}

Provide:
1. Root cause analysis
2. Location of the bug
3. Why it occurred
4. How to fix it

Respond with JSON:
{{
    "root_cause": "analysis",
    "bug_location": "line or function",
    "explanation": "why it happened",
    "fix_suggestion": "how to fix"
}}"""

        if self._model_manager:
            try:
                response = await self._model_manager.generate(prompt)
                
                content = response.content
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    return json.loads(content[json_start:json_end])
            
            except Exception as e:
                self.logger.error(f"Error analysis failed: {e}")
        
        return {'root_cause': 'unknown'}
    
    async def fix_bug(
        self,
        code: str,
        error_message: str,
        stack_trace: Optional[str] = None,
        max_attempts: int = 3
    ) -> Dict[str, Any]:
        """
        Attempt to automatically fix a bug.
        
        Args:
            code: Source code
            error_message: Error message
            stack_trace: Optional stack trace
            max_attempts: Maximum fix attempts
        
        Returns:
            Fix result with corrected code
        """
        # First analyze the error
        diagnosis = await self.analyze_error(code, error_message, stack_trace)
        
        for attempt in range(max_attempts):
            prompt = f"""Fix this Python bug:

Code with error:
```python
{code}
```

Error: {error_message}
Diagnosis: {json.dumps(diagnosis, indent=2)}

Provide the complete fixed code. The code must run without errors.

Respond with JSON:
{{
    "fixed_code": "the corrected code",
    "changes_made": ["list of changes"],
    "explanation": "how the fix works"
}}"""

            if self._model_manager:
                try:
                    response = await self._model_manager.generate(prompt)
                    
                    content = response.content
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        result = json.loads(content[json_start:json_end])
                        
                        # Store in history
                        self._bug_history.append({
                            'error': error_message,
                            'diagnosis': diagnosis,
                            'fix': result.get('changes_made', []),
                            'attempt': attempt + 1,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        return result
                
                except Exception as e:
                    self.logger.error(f"Fix attempt {attempt + 1} failed: {e}")
        
        return {'error': 'Could not fix bug', 'attempts': max_attempts}
    
    async def review_code(self, code: str) -> Dict[str, Any]:
        """
        Review code for potential bugs.
        
        Args:
            code: Source code to review
        
        Returns:
            Code review results
        """
        prompt = f"""Review this Python code for potential bugs and issues:

```python
{code}
```

Look for:
1. Logic errors
2. Edge cases not handled
3. Potential runtime errors
4. Performance issues
5. Security vulnerabilities

Respond with JSON:
{{
    "issues": [
        {{
            "type": "bug_type",
            "location": "line or function",
            "description": "issue description",
            "severity": "low/medium/high",
            "suggested_fix": "how to fix"
        }}
    ],
    "overall_quality": "good/moderate/poor",
    "recommendations": []
}}"""

        if self._model_manager:
            try:
                response = await self._model_manager.generate(prompt)
                
                content = response.content
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    return json.loads(content[json_start:json_end])
            
            except Exception as e:
                self.logger.error(f"Code review failed: {e}")
        
        return {'issues': [], 'overall_quality': 'unknown'}
    
    async def suggest_improvements(self, code: str) -> List[Dict]:
        """Suggest code improvements."""
        review = await self.review_code(code)
        return review.get('recommendations', [])
    
    def get_bug_history(self) -> List[Dict]:
        """Get bug fix history."""
        return self._bug_history
    
    def get_common_patterns(self) -> Dict[str, int]:
        """Get common bug patterns from history."""
        patterns = {}
        
        for entry in self._bug_history:
            diagnosis = entry.get('diagnosis', {})
            root_cause = diagnosis.get('root_cause', 'unknown')
            patterns[root_cause] = patterns.get(root_cause, 0) + 1
        
        return patterns
