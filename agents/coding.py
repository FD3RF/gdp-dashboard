"""
Coding Agent for code generation and modification.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from agents.base_agent import BaseAgent, AgentTask


class CodingAgent(BaseAgent):
    """
    Coding Agent responsible for:
    - Generating trading code
    - Modifying existing strategies
    - Debugging and fixing code issues
    """
    
    def __init__(
        self,
        name: str = 'coding',
        config: Optional[Dict[str, Any]] = None,
        model_manager = None,
        vector_memory = None
    ):
        super().__init__(name, config, model_manager, vector_memory)
        self._generated_code: Dict[str, Dict] = {}
    
    def _get_default_system_prompt(self) -> str:
        return """You are the Coding Agent in a quantitative trading system.
Your role is to:
1. Generate clean, efficient Python code for trading strategies
2. Implement technical indicators and analysis functions
3. Debug and fix code issues
4. Optimize code performance

Follow these guidelines:
- Use type hints
- Include docstrings
- Follow PEP 8 style
- Include error handling
- Write modular, reusable code"""

    async def process_task(self, task: AgentTask) -> Any:
        """Process a coding task."""
        if task.type == 'generate_code':
            return await self._generate_code(task)
        elif task.type == 'modify_code':
            return await self._modify_code(task)
        elif task.type == 'debug_code':
            return await self._debug_code(task)
        elif task.type == 'generate_indicator':
            return await self._generate_indicator(task)
        elif task.type == 'generate_strategy_class':
            return await self._generate_strategy_class(task)
        else:
            raise ValueError(f"Unknown task type: {task.type}")
    
    async def _generate_code(self, task: AgentTask) -> Dict[str, Any]:
        """Generate Python code from specification."""
        specification = task.parameters.get('specification', '')
        language = task.parameters.get('language', 'python')
        context = task.parameters.get('context', {})
        
        prompt = f"""Generate Python code for:

Specification: {specification}

Context: {json.dumps(context, indent=2, default=str)}

Requirements:
1. Clean, readable code
2. Proper error handling
3. Type hints
4. Docstrings
5. Unit test examples

Provide the complete code implementation."""

        response = await self.generate_response(prompt)
        
        # Extract code from response
        code = self._extract_code(response)
        
        code_id = f"code_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self._generated_code[code_id] = {
            'specification': specification,
            'code': code,
            'created_at': datetime.now().isoformat()
        }
        
        return {
            'code_id': code_id,
            'code': code,
            'language': language
        }
    
    async def _modify_code(self, task: AgentTask) -> Dict[str, Any]:
        """Modify existing code."""
        code_id = task.parameters.get('code_id')
        modifications = task.parameters.get('modifications', '')
        
        existing = self._generated_code.get(code_id)
        if not existing:
            # Use provided code directly
            existing_code = task.parameters.get('code', '')
        else:
            existing_code = existing['code']
        
        prompt = f"""Modify this Python code according to the requirements:

Current Code:
```python
{existing_code}
```

Modifications Required: {modifications}

Provide the complete modified code."""

        response = await self.generate_response(prompt)
        modified_code = self._extract_code(response)
        
        if code_id and code_id in self._generated_code:
            self._generated_code[code_id]['code'] = modified_code
            self._generated_code[code_id]['modified_at'] = datetime.now().isoformat()
        
        return {
            'code_id': code_id,
            'modified_code': modified_code
        }
    
    async def _debug_code(self, task: AgentTask) -> Dict[str, Any]:
        """Debug and fix code issues."""
        code = task.parameters.get('code', '')
        error_message = task.parameters.get('error', '')
        context = task.parameters.get('context', '')
        
        prompt = f"""Debug and fix this Python code:

Code:
```python
{code}
```

Error: {error_message}
Context: {context}

Provide:
1. Analysis of the issue
2. Fixed code
3. Explanation of the fix

Respond with JSON:
{{
    "issue_analysis": "what was wrong",
    "fixed_code": "the corrected code",
    "explanation": "how it was fixed"
}}"""

        response = await self.generate_response(prompt)
        
        try:
            json_str = response[response.find('{'):response.rfind('}')+1]
            result = json.loads(json_str)
            result['fixed_code'] = self._extract_code(result.get('fixed_code', code))
            return result
        except Exception as e:
            # Fallback: return extracted code
            return {
                'issue_analysis': 'Could not parse response',
                'fixed_code': self._extract_code(response),
                'explanation': str(e)
            }
    
    async def _generate_indicator(self, task: AgentTask) -> Dict[str, Any]:
        """Generate a technical indicator implementation."""
        indicator_name = task.parameters.get('name', '')
        indicator_type = task.parameters.get('type', 'trend')
        parameters = task.parameters.get('parameters', {})
        
        prompt = f"""Implement a Python function for the {indicator_name} indicator.

Type: {indicator_type}
Parameters: {json.dumps(parameters, indent=2)}

Requirements:
1. Accept a pandas DataFrame with OHLCV data
2. Return the indicator values
3. Handle edge cases
4. Include usage example

Provide complete implementation with type hints."""

        response = await self.generate_response(prompt)
        code = self._extract_code(response)
        
        return {
            'indicator_name': indicator_name,
            'indicator_type': indicator_type,
            'code': code
        }
    
    async def _generate_strategy_class(self, task: AgentTask) -> Dict[str, Any]:
        """Generate a complete strategy class."""
        strategy_config = task.parameters.get('config', {})
        base_class = task.parameters.get('base_class', 'BaseStrategy')
        
        prompt = f"""Generate a complete trading strategy class:

Configuration: {json.dumps(strategy_config, indent=2)}
Base Class: {base_class}

Include:
1. All required methods (initialize, on_data, on_order, etc.)
2. Entry/exit logic
3. Risk management integration
4. Parameter validation

Provide the complete class implementation."""

        response = await self.generate_response(prompt)
        code = self._extract_code(response)
        
        return {
            'strategy_name': strategy_config.get('name', 'GeneratedStrategy'),
            'code': code,
            'base_class': base_class
        }
    
    def _extract_code(self, text: str) -> str:
        """Extract code from markdown or plain text."""
        # Try to find code blocks
        code_pattern = r'```(?:python)?\s*\n(.*?)```'
        matches = re.findall(code_pattern, text, re.DOTALL)
        
        if matches:
            return '\n\n'.join(matches)
        
        # No code blocks, return the text as-is
        return text
    
    def get_generated_code(self, code_id: str) -> Optional[Dict]:
        """Get generated code by ID."""
        return self._generated_code.get(code_id)
    
    def list_generated_code(self) -> List[Dict]:
        """List all generated code."""
        return [
            {
                'code_id': cid,
                'specification': c['specification'][:100],
                'created_at': c['created_at']
            }
            for cid, c in self._generated_code.items()
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """Get coding agent status."""
        return {
            **super().get_status(),
            'generated_code_count': len(self._generated_code)
        }
