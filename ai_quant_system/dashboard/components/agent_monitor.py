"""
Agent Monitor Component
=======================

Real-time AI agent monitoring panel.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class AgentMonitor:
    """
    AI Agent monitoring dashboard component.
    """
    
    def __init__(self, api_base: str = "http://localhost:8000"):
        self.api_base = api_base
    
    def render(self, agents: Dict[str, Any]) -> None:
        """
        Render the agent monitoring panel.
        
        Args:
            agents: Dictionary of agent instances or status data
        """
        st.header("🤖 AI Agent Monitor")
        
        # Agent overview
        self._render_overview(agents)
        
        # Individual agent cards
        st.subheader("Agent Details")
        
        # Create tabs for each agent
        agent_names = list(agents.keys()) if agents else []
        
        if agent_names:
            tabs = st.tabs(agent_names)
            
            for i, (name, agent_data) in enumerate(agents.items()):
                with tabs[i]:
                    self._render_agent_card(name, agent_data)
        else:
            st.info("No agents configured")
    
    def _render_overview(self, agents: Dict[str, Any]) -> None:
        """Render agent overview metrics."""
        col1, col2, col3, col4 = st.columns(4)
        
        total_agents = len(agents)
        running_agents = sum(
            1 for a in agents.values() 
            if self._get_agent_status(a) == 'running'
        )
        total_tasks = sum(
            a.get('queued_tasks', 0) 
            for a in agents.values() 
            if isinstance(a, dict)
        )
        
        with col1:
            st.metric("Total Agents", total_agents)
        
        with col2:
            st.metric("Running", running_agents)
        
        with col3:
            st.metric("Idle", total_agents - running_agents)
        
        with col4:
            st.metric("Queued Tasks", total_tasks)
        
        # Agent status visualization
        self._render_status_chart(agents)
    
    def _render_status_chart(self, agents: Dict[str, Any]) -> None:
        """Render agent status bar chart."""
        if not agents:
            return
        
        status_counts = {
            'Running': 0,
            'Idle': 0,
            'Error': 0
        }
        
        for agent in agents.values():
            status = self._get_agent_status(agent)
            if status == 'running':
                status_counts['Running'] += 1
            elif status == 'error':
                status_counts['Error'] += 1
            else:
                status_counts['Idle'] += 1
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(status_counts.keys()),
                y=list(status_counts.values()),
                marker_color=['#4CAF50', '#2196F3', '#f44336']
            )
        ])
        
        fig.update_layout(
            title="Agent Status Distribution",
            template='plotly_dark',
            height=200,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_agent_card(self, name: str, agent_data: Any) -> None:
        """Render individual agent card."""
        # Get status info
        if isinstance(agent_data, dict):
            status_info = agent_data
        else:
            status_info = self._extract_agent_info(agent_data)
        
        # Status badge
        status = self._get_agent_status(agent_data)
        status_emoji = {
            'running': '🟢',
            'idle': '🟡',
            'error': '🔴',
            'stopped': '⚪'
        }.get(status, '⚪')
        
        st.markdown(f"### {status_emoji} {name}")
        
        # Agent metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Status",
                status.upper()
            )
        
        with col2:
            st.metric(
                "Queued Tasks",
                status_info.get('queued_tasks', 0)
            )
        
        with col3:
            st.metric(
                "Uptime",
                status_info.get('uptime', 'N/A')
            )
        
        # Recent tasks
        st.subheader("Recent Tasks")
        recent_tasks = status_info.get('recent_tasks', [])
        
        if recent_tasks:
            df = pd.DataFrame(recent_tasks)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No recent tasks")
        
        # Task statistics
        if 'task_stats' in status_info:
            self._render_task_stats(status_info['task_stats'])
    
    def _render_task_stats(self, stats: Dict[str, Any]) -> None:
        """Render task statistics."""
        st.subheader("Task Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Completed", stats.get('completed', 0))
        
        with col2:
            st.metric("Failed", stats.get('failed', 0))
        
        with col3:
            st.metric("Success Rate", f"{stats.get('success_rate', 0):.1f}%")
        
        with col4:
            st.metric("Avg Duration", f"{stats.get('avg_duration', 0):.2f}s")
    
    def _get_agent_status(self, agent: Any) -> str:
        """Get agent status string."""
        if isinstance(agent, dict):
            return agent.get('status', 'unknown')
        
        if hasattr(agent, 'status'):
            return agent.status.value if hasattr(agent.status, 'value') else str(agent.status)
        
        if hasattr(agent, 'is_running'):
            return 'running' if agent.is_running else 'stopped'
        
        return 'unknown'
    
    def _extract_agent_info(self, agent: Any) -> Dict[str, Any]:
        """Extract agent info from object."""
        info = {}
        
        if hasattr(agent, 'get_status'):
            info = agent.get_status()
        elif hasattr(agent, 'status'):
            info['status'] = self._get_agent_status(agent)
        
        if hasattr(agent, '_task_queue'):
            info['queued_tasks'] = len(agent._task_queue)
        
        if hasattr(agent, 'uptime'):
            info['uptime'] = str(agent.uptime)
        
        return info


def render_agent_activity_timeline(
    activities: List[Dict[str, Any]]
) -> go.Figure:
    """
    Render agent activity timeline.
    
    Args:
        activities: List of activity records
    
    Returns:
        Plotly Figure
    """
    if not activities:
        return go.Figure()
    
    df = pd.DataFrame(activities)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig = go.Figure()
    
    # Add scatter for each agent
    agents = df['agent'].unique() if 'agent' in df.columns else ['Agent']
    
    for i, agent in enumerate(agents):
        agent_df = df[df['agent'] == agent] if 'agent' in df.columns else df
        
        fig.add_trace(
            go.Scatter(
                x=agent_df['timestamp'] if 'timestamp' in agent_df.columns else list(range(len(agent_df))),
                y=[i] * len(agent_df),
                mode='markers',
                name=agent,
                marker=dict(
                    size=10,
                    symbol='circle'
                ),
                hovertemplate=(
                    f"<b>{agent}</b><br>"
                    "Task: %{text}<br>"
                    "<extra></extra>"
                ),
                text=agent_df['task'] if 'task' in agent_df.columns else ['N/A'] * len(agent_df)
            )
        )
    
    fig.update_layout(
        title="Agent Activity Timeline",
        template='plotly_dark',
        height=300,
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(agents))),
            ticktext=list(agents)
        ),
        xaxis_title='Time',
        yaxis_title='Agent'
    )
    
    return fig


def render_agent_task_distribution(
    tasks: List[Dict[str, Any]]
) -> go.Figure:
    """
    Render task distribution pie chart.
    
    Args:
        tasks: List of task records
    
    Returns:
        Plotly Figure
    """
    if not tasks:
        return go.Figure()
    
    df = pd.DataFrame(tasks)
    
    if 'agent' not in df.columns:
        return go.Figure()
    
    task_counts = df['agent'].value_counts()
    
    fig = go.Figure(
        data=[
            go.Pie(
                labels=task_counts.index,
                values=task_counts.values,
                hole=0.4,
                textinfo='label+percent',
                textposition='inside'
            )
        ]
    )
    
    fig.update_layout(
        title="Task Distribution by Agent",
        template='plotly_dark',
        height=300
    )
    
    return fig
