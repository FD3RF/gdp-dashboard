"""
Auto Code Refactor for automated code improvement.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from core.base import BaseModule


class AutoCodeRefactor(BaseModule):
    """
    Automatically refactors and improves code.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('auto_code_refactor', config)
        
        self._model_manager = None
        self._refactor_history: List[Dict] = []
    
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
    
    async def refactor(
        self,
        code: str,
        goals: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Refactor code with specified goals.
        
        Args:
            code: Source code to refactor
            goals: Refactoring goals
        
        Returns:
            Refactored code and changes
        """
        goals = goals or ['readability', 'performance', 'maintainability']
        
        prompt = f"""Refactor this Python code with the following goals: {goals}

Original Code:
```python
{code}
```

Provide:
1. Refactored code
2. List of changes made
3. Expected improvements

Respond with JSON:
{{
    "refactored_code": "the improved code",
    "changes": ["list of changes"],
    "improvements": ["expected improvements"]
}}"""

        if self._model_manager:
            try:
                response = await self._model_manager.generate(prompt)
                
                content = response.content
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    result = json.loads(content[json_start:json_end])
                    
                    self._refactor_history.append({
                        'original_code': code[:500],  # Store first 500 chars
                        'changes': result.get('changes', []),
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    return result
            
            except Exception as e:
                self.logger.error(f"Refactoring error: {e}")
        
        return {'refactored_code': code, 'changes': [], 'error': 'Refactoring failed'}
    
    async def optimize_performance(self, code: str) -> Dict[str, Any]:
        """Optimize code for performance."""
        return await self.refactor(code, ['performance', 'efficiency'])
    
    async def improve_readability(self, code: str) -> Dict[str, Any]:
        """Improve code readability."""
        return await self.refactor(code, ['readability', 'documentation'])
    
    async def add_type_hints(self, code: str) -> Dict[str, Any]:
        """Add type hints to code."""
        return await self.refactor(code, ['type_safety', 'documentation'])
    
    async def add_error_handling(self, code: str) -> Dict[str, Any]:
        """Add error handling to code."""
        return await self.refactor(code, ['error_handling', 'robustness'])
    
    async def convert_to_async(self, code: str) -> Dict[str, Any]:
        """Convert synchronous code to async."""
        prompt = f"""Convert this synchronous Python code to async:

```python
{code}
```

Maintain the same functionality but use async/await patterns.
Add appropriate imports and handle async context properly.

Provide the complete refactored code."""

        if self._model_manager:
            try:
                response = await self._model_manager.generate(prompt)
                return {'refactored_code': response.content}
            except Exception as e:
                return {'error': str(e)}
        
        return {'refactored_code': code}
    
    def get_refactor_history(self) -> List[Dict]:
        """Get refactoring history."""
        return self._refactor_history
