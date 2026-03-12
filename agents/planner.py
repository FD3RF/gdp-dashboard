"""
Planner Agent for task planning and coordination.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from agents.base_agent import BaseAgent, AgentTask, AgentState


class PlannerAgent(BaseAgent):
    """
    Planner Agent responsible for:
    - Breaking down complex tasks into subtasks
    - Coordinating other agents
    - Managing workflow execution
    """
    
    def __init__(
        self,
        name: str = 'planner',
        config: Optional[Dict[str, Any]] = None,
        model_manager = None,
        vector_memory = None
    ):
        super().__init__(name, config, model_manager, vector_memory)
        self._agents: Dict[str, BaseAgent] = {}
        self._plan_history: List[Dict] = []
    
    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent for coordination."""
        self._agents[agent.name] = agent
        self.logger.info(f"Registered agent: {agent.name}")
    
    def unregister_agent(self, agent_name: str) -> None:
        """Unregister an agent."""
        if agent_name in self._agents:
            del self._agents[agent_name]
    
    def _get_default_system_prompt(self) -> str:
        return """You are the Planner Agent in a quantitative trading system.
Your role is to:
1. Analyze complex tasks and break them into subtasks
2. Determine which agent should handle each subtask
3. Create execution plans with proper sequencing
4. Monitor and adjust plans as needed

Available agents:
- ResearchAgent: Market research and analysis
- StrategyAgent: Strategy generation and management
- CodingAgent: Code generation and modification
- BacktestAgent: Strategy backtesting
- RiskAgent: Risk analysis and management
- ExecutionAgent: Trade execution

Respond with JSON-formatted plans."""

    async def process_task(self, task: AgentTask) -> Any:
        """Process a planning task."""
        if task.type == 'create_plan':
            return await self._create_plan(task)
        elif task.type == 'execute_plan':
            return await self._execute_plan(task)
        elif task.type == 'analyze_request':
            return await self._analyze_request(task)
        else:
            raise ValueError(f"Unknown task type: {task.type}")
    
    async def _analyze_request(self, task: AgentTask) -> Dict[str, Any]:
        """Analyze a user request and determine required actions."""
        request = task.parameters.get('request', '')
        
        prompt = f"""Analyze this request and determine what needs to be done:
        
Request: {request}

Respond with JSON containing:
{{
    "intent": "brief description of intent",
    "complexity": "low/medium/high",
    "required_agents": ["list of agents needed"],
    "estimated_steps": number,
    "dependencies": ["any dependencies between steps"]
}}"""

        response = await self.generate_response(prompt)
        
        try:
            # Extract JSON from response
            json_str = response[response.find('{'):response.rfind('}')+1]
            analysis = json.loads(json_str)
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            self.logger.warning(f"Failed to parse analysis JSON: {e}, using default")
            analysis = {
                'intent': request,
                'complexity': 'medium',
                'required_agents': ['research'],
                'estimated_steps': 3
            }
        
        return analysis
    
    async def _create_plan(self, task: AgentTask) -> Dict[str, Any]:
        """Create an execution plan for a goal."""
        goal = task.parameters.get('goal', '')
        context = task.parameters.get('context', {})
        
        prompt = f"""Create a detailed execution plan for this goal:

Goal: {goal}

Context: {json.dumps(context, indent=2)}

Available Agents: {list(self._agents.keys())}

Create a plan with specific steps. Respond with JSON:
{{
    "plan_id": "unique_id",
    "goal": "the goal",
    "steps": [
        {{
            "step_number": 1,
            "agent": "agent_name",
            "task_type": "task_type",
            "description": "what to do",
            "parameters": {{}},
            "dependencies": []
        }}
    ],
    "estimated_duration": "time estimate"
}}"""

        response = await self.generate_response(prompt)
        
        try:
            json_str = response[response.find('{'):response.rfind('}')+1]
            plan = json.loads(json_str)
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            self.logger.warning(f"Failed to parse plan JSON: {e}, creating default plan")
            # Create default plan
            plan = {
                'plan_id': f"plan_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'goal': goal,
                'steps': [
                    {
                        'step_number': 1,
                        'agent': 'research',
                        'task_type': 'analyze',
                        'description': f'Research: {goal}',
                        'parameters': context,
                        'dependencies': []
                    }
                ]
            }
        
        self._plan_history.append({
            'plan': plan,
            'created_at': datetime.now().isoformat(),
            'status': 'created'
        })
        
        return plan
    
    async def _execute_plan(self, task: AgentTask) -> Dict[str, Any]:
        """Execute a plan step by step."""
        plan = task.parameters.get('plan', {})
        steps = plan.get('steps', [])
        
        results = []
        completed_steps = set()
        
        for step in steps:
            step_num = step['step_number']
            agent_name = step['agent']
            
            # Check dependencies
            deps = step.get('dependencies', [])
            if not all(d in completed_steps for d in deps):
                self.logger.warning(f"Step {step_num} dependencies not met, skipping")
                continue
            
            # Get agent
            agent = self._agents.get(agent_name)
            if not agent:
                self.logger.error(f"Agent not found: {agent_name}")
                results.append({
                    'step': step_num,
                    'status': 'failed',
                    'error': f'Agent {agent_name} not found'
                })
                continue
            
            # Execute step
            try:
                task_id = await agent.submit_task(
                    task_type=step['task_type'],
                    description=step['description'],
                    parameters=step.get('parameters', {})
                )
                
                result = await agent.run_task(task_id)
                
                results.append({
                    'step': step_num,
                    'agent': agent_name,
                    'status': 'completed',
                    'result': str(result)[:500]
                })
                
                completed_steps.add(step_num)
                
            except Exception as e:
                results.append({
                    'step': step_num,
                    'agent': agent_name,
                    'status': 'failed',
                    'error': str(e)
                })
        
        return {
            'plan_id': plan.get('plan_id'),
            'results': results,
            'completed': len(completed_steps),
            'total': len(steps)
        }
    
    async def decompose_task(
        self,
        task_description: str,
        max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """Decompose a complex task into subtasks."""
        prompt = f"""Decompose this task into smaller subtasks:

Task: {task_description}

Available Agents: {list(self._agents.keys())}

Create a hierarchical decomposition (max depth {max_depth}). Respond with JSON:
{{
    "subtasks": [
        {{
            "id": "subtask_1",
            "description": "description",
            "agent": "recommended agent",
            "priority": 1-10,
            "subtasks": []
        }}
    ]
}}"""

        response = await self.generate_response(prompt)
        
        try:
            json_str = response[response.find('{'):response.rfind('}')+1]
            decomposition = json.loads(json_str)
            return decomposition.get('subtasks', [])
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            self.logger.warning(f"Failed to parse decomposition JSON: {e}")
            return []
    
    def get_plan_history(self) -> List[Dict]:
        """Get plan execution history."""
        return self._plan_history
    
    def get_status(self) -> Dict[str, Any]:
        """Get planner status."""
        return {
            **super().get_status(),
            'registered_agents': list(self._agents.keys()),
            'plans_created': len(self._plan_history)
        }
