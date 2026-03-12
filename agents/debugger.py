"""
Debugger Agent
==============

Analyzes and fixes errors with retry limits to prevent infinite loops.
Works with CodeReviewer for pre-execution validation.
"""

import re
import logging
import traceback
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from core.base import BaseModule
from core.memory import MemoryStore, MemoryType
from agents.code_reviewer import CodeReviewer, IssueSeverity


class FixStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    MAX_RETRY = "max_retry_exceeded"
    NEEDS_HUMAN = "needs_human_intervention"


@dataclass
class DebugSession:
    """Tracks a debugging session."""
    session_id: str
    original_error: str
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    status: FixStatus = FixStatus.FAILED
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None


class DebuggerAgent(BaseModule):
    """
    AI Debugger Agent with retry limits.
    
    Features:
    - Error analysis and categorization
    - Automatic fix suggestions
    - Retry limit to prevent infinite loops
    - Integration with CodeReviewer
    - Memory of past errors and solutions
    """
    
    MAX_RETRY = 5  # Maximum fix attempts
    MAX_SIMILAR_ERRORS = 3  # Stop if same error repeats
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('debugger', config)
        
        self.memory = MemoryStore()
        self.code_reviewer = CodeReviewer()
        
        # Error patterns and their solutions
        self.error_solutions = {
            # Import errors
            r"ModuleNotFoundError: No module named '(\w+)'": {
                'type': 'import',
                'solution': 'Install missing module or fix import path',
                'action': 'pip install {module}'
            },
            r"ImportError: cannot import name '(\w+)'": {
                'type': 'import',
                'solution': 'Check if the name exists in the module',
            },
            
            # Syntax errors
            r"SyntaxError: (.+)": {
                'type': 'syntax',
                'solution': 'Fix syntax error in the code',
            },
            r"IndentationError: (.+)": {
                'type': 'syntax',
                'solution': 'Fix indentation (use 4 spaces)',
            },
            
            # Type errors
            r"TypeError: (.+)": {
                'type': 'type',
                'solution': 'Check types and add type conversion if needed',
            },
            r"AttributeError: '(\w+)' object has no attribute '(\w+)'": {
                'type': 'attribute',
                'solution': "Check if attribute exists or object is correct type",
            },
            
            # Name errors
            r"NameError: name '(\w+)' is not defined": {
                'type': 'name',
                'solution': "Define the variable or import the module",
            },
            
            # Key/Index errors
            r"KeyError: (.+)": {
                'type': 'key',
                'solution': "Check if key exists in dictionary before accessing",
            },
            r"IndexError: (.+)": {
                'type': 'index',
                'solution': "Check list/array bounds before accessing",
            },
            
            # File errors
            r"FileNotFoundError: (.+)": {
                'type': 'file',
                'solution': "Create file or check file path",
            },
            r"PermissionError: (.+)": {
                'type': 'permission',
                'solution': "Check file permissions",
            },
            
            # Async errors
            r"RuntimeError: (.+was never awaited.+)": {
                'type': 'async',
                'solution': "Add await keyword before async function call",
            },
        }
    
    async def initialize(self) -> bool:
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        self._running = True
        return True
    
    async def stop(self) -> bool:
        self._running = False
        return True
    
    def analyze_error(self, error: str, traceback_str: str = None) -> Dict[str, Any]:
        """
        Analyze an error and provide categorization and suggestions.
        
        Args:
            error: Error message
            traceback_str: Full traceback (optional)
            
        Returns:
            Analysis result with suggestions
        """
        analysis = {
            'error': error,
            'type': 'unknown',
            'solution': None,
            'action': None,
            'file': None,
            'line': None,
            'suggestions': []
        }
        
        # Extract file and line from traceback
        if traceback_str:
            file_match = re.search(r'File "([^"]+)", line (\d+)', traceback_str)
            if file_match:
                analysis['file'] = file_match.group(1)
                analysis['line'] = int(file_match.group(2))
        
        # Match against known patterns
        for pattern, info in self.error_solutions.items():
            match = re.search(pattern, error)
            if match:
                analysis['type'] = info['type']
                analysis['solution'] = info['solution']
                
                if 'action' in info:
                    analysis['action'] = info['action'].format(
                        module=match.group(1) if match.groups() else ''
                    )
                
                analysis['matched_groups'] = match.groups()
                break
        
        # Add context-aware suggestions
        if analysis['type'] == 'import':
            analysis['suggestions'].append("Check requirements.txt for missing dependencies")
            analysis['suggestions'].append("Verify the import path is correct")
        
        elif analysis['type'] == 'name':
            analysis['suggestions'].append("Check for typos in variable/function name")
            analysis['suggestions'].append("Add import if it's from an external module")
            analysis['suggestions'].append("Define the variable before use")
        
        elif analysis['type'] == 'syntax':
            analysis['suggestions'].append("Check for missing brackets, quotes, or colons")
            analysis['suggestions'].append("Verify indentation is consistent")
        
        elif analysis['type'] == 'async':
            analysis['suggestions'].append("Ensure async functions are called with await")
            analysis['suggestions'].append("Check if running in async context")
        
        # Remember this error
        self.memory.remember_error(error, {
            'type': analysis['type'],
            'file': analysis['file'],
            'line': analysis['line']
        })
        
        return analysis
    
    async def debug_session(self, error: str, traceback_str: str = None,
                           fix_callback=None) -> DebugSession:
        """
        Run a complete debugging session with retry limits.
        
        Args:
            error: Initial error message
            traceback_str: Full traceback
            fix_callback: Async function to apply fixes
            
        Returns:
            DebugSession with results
        """
        session = DebugSession(
            session_id=f"debug_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            original_error=error
        )
        
        current_error = error
        similar_error_count = {}
        
        while len(session.attempts) < self.MAX_RETRY:
            attempt_num = len(session.attempts) + 1
            self.logger.info(f"Debug attempt {attempt_num}/{self.MAX_RETRY}")
            
            # Analyze error
            analysis = self.analyze_error(current_error, traceback_str)
            
            # Track similar errors
            error_key = analysis['type'] + ':' + (analysis['file'] or 'unknown')
            similar_error_count[error_key] = similar_error_count.get(error_key, 0) + 1
            
            # Check if same error repeats too many times
            if similar_error_count[error_key] >= self.MAX_SIMILAR_ERRORS:
                session.status = FixStatus.MAX_RETRY
                session.attempts.append({
                    'attempt': attempt_num,
                    'error': current_error,
                    'analysis': analysis,
                    'result': 'similar_error_repeated',
                    'suggestion': 'Manual intervention required - same error keeps occurring'
                })
                break
            
            # Record attempt
            attempt = {
                'attempt': attempt_num,
                'error': current_error,
                'analysis': analysis
            }
            
            # If fix callback provided, try to fix
            if fix_callback:
                try:
                    fix_result = await fix_callback(analysis)
                    attempt['fix_result'] = fix_result
                    
                    if fix_result.get('success'):
                        # Run code review before considering success
                        if fix_result.get('file'):
                            issues = await self.code_reviewer.review_file(
                                fix_result['file'],
                                fix_result.get('new_content', '')
                            )
                            
                            critical_issues = [
                                i for i in issues 
                                if i.severity == IssueSeverity.ERROR
                            ]
                            
                            if not critical_issues:
                                session.status = FixStatus.SUCCESS
                                attempt['result'] = 'fixed'
                            else:
                                attempt['result'] = 'fix_has_issues'
                                attempt['remaining_issues'] = [
                                    i.to_dict() for i in critical_issues
                                ]
                                current_error = f"Code review failed: {critical_issues[0].message}"
                        else:
                            session.status = FixStatus.SUCCESS
                            attempt['result'] = 'fixed'
                    
                    else:
                        attempt['result'] = 'fix_failed'
                        current_error = fix_result.get('error', 'Unknown fix error')
                
                except Exception as e:
                    attempt['result'] = 'fix_exception'
                    attempt['exception'] = str(e)
                    current_error = str(e)
            
            else:
                # No fix callback - just analysis
                attempt['result'] = 'analysis_only'
                break
            
            session.attempts.append(attempt)
            
            # Check if resolved
            if session.status == FixStatus.SUCCESS:
                break
        
        # Finalize session
        session.end_time = datetime.now().isoformat()
        
        if len(session.attempts) >= self.MAX_RETRY and session.status != FixStatus.SUCCESS:
            session.status = FixStatus.NEEDS_HUMAN
            self.logger.warning(f"Debug session {session.session_id} needs human intervention")
        
        # Store in memory
        self.memory.remember(
            MemoryType.ERROR_LOG,
            {
                'session_id': session.session_id,
                'original_error': session.original_error,
                'attempts': len(session.attempts),
                'status': session.status.value
            },
            importance=0.9
        )
        
        return session
    
    def get_fix_suggestion(self, analysis: Dict[str, Any], 
                          file_content: str = None) -> str:
        """
        Generate a detailed fix suggestion for AI to implement.
        
        Args:
            analysis: Error analysis result
            file_content: Content of the file with error (optional)
            
        Returns:
            Detailed fix suggestion prompt
        """
        suggestion = f"""
## Error Analysis

**Type:** {analysis['type']}
**Error:** {analysis['error']}

"""
        
        if analysis.get('file'):
            suggestion += f"**File:** {analysis['file']}"
            if analysis.get('line'):
                suggestion += f" (line {analysis['line']})"
            suggestion += "\n\n"
        
        if analysis.get('solution'):
            suggestion += f"### Solution\n{analysis['solution']}\n\n"
        
        if analysis.get('suggestions'):
            suggestion += "### Suggestions:\n"
            for s in analysis['suggestions']:
                suggestion += f"- {s}\n"
        
        if file_content and analysis.get('line'):
            # Show context around error
            lines = file_content.split('\n')
            start = max(0, analysis['line'] - 5)
            end = min(len(lines), analysis['line'] + 5)
            
            suggestion += "\n### Code Context:\n```\n"
            for i in range(start, end):
                marker = ">>>" if i == analysis['line'] - 1 else "   "
                suggestion += f"{marker} {i+1:4d} | {lines[i]}\n"
            suggestion += "```\n"
        
        return suggestion
    
    def should_escalate(self, session: DebugSession) -> bool:
        """Check if debugging should be escalated to human."""
        return session.status in [FixStatus.MAX_RETRY, FixStatus.NEEDS_HUMAN]
    
    def get_debug_report(self, session: DebugSession) -> str:
        """Generate a human-readable debug report."""
        report = f"""
# Debug Report: {session.session_id}

**Status:** {session.status.value}
**Original Error:** {session.original_error}
**Attempts:** {len(session.attempts)}/{self.MAX_RETRY}
**Start Time:** {session.start_time}
**End Time:** {session.end_time or 'N/A'}

## Attempts

"""
        for attempt in session.attempts:
            report += f"### Attempt {attempt['attempt']}\n"
            report += f"- **Error:** {attempt['error'][:100]}...\n"
            report += f"- **Type:** {attempt['analysis']['type']}\n"
            report += f"- **Result:** {attempt['result']}\n\n"
        
        if self.should_escalate(session):
            report += "\n⚠️ **This issue requires human intervention.**\n"
        
        return report


# Export
__all__ = ['DebuggerAgent', 'DebugSession', 'FixStatus']
