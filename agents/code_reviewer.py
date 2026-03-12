"""
Code Review Agent
=================

Reviews code before execution to catch bugs, syntax errors, and import issues.
Acts as a quality gate between Coding and Runner agents.
"""

import ast
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from core.base import BaseModule
from core.memory import MemoryStore, MemoryType


class IssueSeverity(Enum):
    ERROR = "error"      # Must fix before running
    WARNING = "warning"  # Should fix
    INFO = "info"        # Suggestion


@dataclass
class CodeIssue:
    """Represents a code issue found during review."""
    file_path: str
    line: int
    severity: IssueSeverity
    message: str
    suggestion: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'file_path': self.file_path,
            'line': self.line,
            'severity': self.severity.value,
            'message': self.message,
            'suggestion': self.suggestion
        }


class CodeReviewer(BaseModule):
    """
    AI Code Review Agent.
    
    Performs static analysis and AI-assisted code review:
    - Syntax validation
    - Import checking
    - Common bug patterns
    - Code quality issues
    - Security vulnerabilities
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('code_reviewer', config)
        self.memory = MemoryStore()
        
        # Common issues to check
        self.checks = {
            'syntax': True,
            'imports': True,
            'undefined_vars': True,
            'common_bugs': True,
            'security': True,
            'style': False  # Optional
        }
        
        # Bug patterns to detect
        self.bug_patterns = [
            # Missing imports
            (r'(?<!from )(?<!import )\b(pd|np|json|os|sys|re|asyncio|logging)\b\.', 
             'Missing import for {match}'),
            # Undefined function calls
            (r'(?<!def )(?<!\.)(\w+)\s*\([^)]*\)\s*:', 
             'Potential undefined function: {match}'),
            # Unused variables
            (r'^\s*(\w+)\s*=\s*[^=].*$', 
             None),  # Complex to detect properly
        ]
        
        # Security patterns
        self.security_patterns = [
            (r'eval\s*\(', 'Use of eval() is dangerous'),
            (r'exec\s*\(', 'Use of exec() is dangerous'),
            (r'__import__\s*\(', 'Dynamic import can be dangerous'),
            (r'subprocess\.(call|run|Popen)\s*\([^)]*shell\s*=\s*True', 
             'Shell=True in subprocess can lead to injection'),
            (r'password\s*=\s*["\'][^"\']+["\']', 
             'Hardcoded password detected'),
        ]
    
    async def initialize(self) -> bool:
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        self._running = True
        return True
    
    async def stop(self) -> bool:
        self._running = False
        return True
    
    async def review_file(self, file_path: str, content: str) -> List[CodeIssue]:
        """
        Review a single file for issues.
        
        Args:
            file_path: Path to the file
            content: File content
            
        Returns:
            List of issues found
        """
        issues = []
        
        # 1. Syntax check
        if self.checks['syntax']:
            syntax_issues = self._check_syntax(file_path, content)
            issues.extend(syntax_issues)
        
        # If syntax is broken, skip other checks
        if any(i.severity == IssueSeverity.ERROR for i in issues):
            return issues
        
        # 2. Import check
        if self.checks['imports']:
            import_issues = self._check_imports(file_path, content)
            issues.extend(import_issues)
        
        # 3. Security check
        if self.checks['security']:
            security_issues = self._check_security(file_path, content)
            issues.extend(security_issues)
        
        # 4. Common bug patterns
        if self.checks['common_bugs']:
            bug_issues = self._check_common_bugs(file_path, content)
            issues.extend(bug_issues)
        
        # Store review in memory
        if issues:
            self.memory.remember(
                MemoryType.ERROR_LOG,
                {
                    'type': 'code_review',
                    'file': file_path,
                    'issues': [i.to_dict() for i in issues]
                },
                importance=0.7
            )
        
        return issues
    
    def _check_syntax(self, file_path: str, content: str) -> List[CodeIssue]:
        """Check Python syntax."""
        issues = []
        
        try:
            ast.parse(content)
        except SyntaxError as e:
            issues.append(CodeIssue(
                file_path=file_path,
                line=e.lineno or 1,
                severity=IssueSeverity.ERROR,
                message=f"Syntax error: {e.msg}",
                suggestion=f"Fix syntax at line {e.lineno}"
            ))
        
        return issues
    
    def _check_imports(self, file_path: str, content: str) -> List[CodeIssue]:
        """Check for import issues."""
        issues = []
        
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return issues
        
        # Get all imports in the file
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
        
        # Get all used names
        used_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)
        
        # Check for potential missing imports
        common_modules = {'pd', 'np', 'json', 'os', 'sys', 're', 'asyncio', 
                         'logging', 'datetime', 'pathlib', 'typing', 'collections'}
        
        # Map aliases to modules
        alias_map = {
            'pd': 'pandas',
            'np': 'numpy',
            'plt': 'matplotlib.pyplot',
            'sns': 'seaborn',
            'tf': 'tensorflow',
            'torch': 'torch'
        }
        
        for name in used_names:
            if name in alias_map and alias_map[name].split('.')[0] not in imports:
                issues.append(CodeIssue(
                    file_path=file_path,
                    line=1,
                    severity=IssueSeverity.ERROR,
                    message=f"Missing import: '{alias_map[name]}' (used as '{name}')",
                    suggestion=f"Add: import {alias_map[name]} as {name}"
                ))
        
        return issues
    
    def _check_security(self, file_path: str, content: str) -> List[CodeIssue]:
        """Check for security issues."""
        issues = []
        lines = content.split('\n')
        
        for pattern, message in self.security_patterns:
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    issues.append(CodeIssue(
                        file_path=file_path,
                        line=i,
                        severity=IssueSeverity.WARNING,
                        message=message,
                        suggestion="Review and fix security issue"
                    ))
        
        return issues
    
    def _check_common_bugs(self, file_path: str, content: str) -> List[CodeIssue]:
        """Check for common bug patterns."""
        issues = []
        lines = content.split('\n')
        
        # Check for common mistakes
        for i, line in enumerate(lines, 1):
            # Mutable default arguments
            if re.search(r'def\s+\w+\([^)]*=\s*\[\]', line):
                issues.append(CodeIssue(
                    file_path=file_path,
                    line=i,
                    severity=IssueSeverity.WARNING,
                    message="Mutable default argument (list)",
                    suggestion="Use None as default and create list inside function"
                ))
            
            if re.search(r'def\s+\w+\([^)]*=\s*\{\}', line):
                issues.append(CodeIssue(
                    file_path=file_path,
                    line=i,
                    severity=IssueSeverity.WARNING,
                    message="Mutable default argument (dict)",
                    suggestion="Use None as default and create dict inside function"
                ))
            
            # Missing await
            if re.search(r'await\s+await\s+', line):
                issues.append(CodeIssue(
                    file_path=file_path,
                    line=i,
                    severity=IssueSeverity.ERROR,
                    message="Double await detected",
                    suggestion="Remove extra await"
                ))
        
        return issues
    
    async def review_project(self, files: Dict[str, str]) -> Dict[str, List[CodeIssue]]:
        """
        Review multiple files in a project.
        
        Args:
            files: Dictionary of {file_path: content}
            
        Returns:
            Dictionary of {file_path: issues}
        """
        results = {}
        
        for file_path, content in files.items():
            if file_path.endswith('.py'):
                issues = await self.review_file(file_path, content)
                if issues:
                    results[file_path] = issues
        
        return results
    
    def get_review_summary(self, issues: Dict[str, List[CodeIssue]]) -> str:
        """Get a summary of review results."""
        total_errors = sum(
            1 for file_issues in issues.values()
            for issue in file_issues
            if issue.severity == IssueSeverity.ERROR
        )
        total_warnings = sum(
            1 for file_issues in issues.values()
            for issue in file_issues
            if issue.severity == IssueSeverity.WARNING
        )
        
        summary = f"Code Review Summary:\n"
        summary += f"- Errors: {total_errors}\n"
        summary += f"- Warnings: {total_warnings}\n"
        summary += f"- Files with issues: {len(issues)}\n"
        
        if total_errors > 0:
            summary += "\n⚠️ Cannot proceed - fix errors first!"
        elif total_warnings > 0:
            summary += "\n✅ Can proceed - but consider fixing warnings"
        else:
            summary += "\n✅ Code looks good!"
        
        return summary


# Export
__all__ = ['CodeReviewer', 'CodeIssue', 'IssueSeverity']
