"""
AI代理监控组件
===============

实时AI代理监控面板，全中文界面。
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# 中文标签
LABELS = {
    'agent_monitor': '🤖 AI代理监控',
    'agent_details': '代理详情',
    'no_agents': '暂无配置代理',
    'total_agents': '代理总数',
    'running': '运行中',
    'idle': '空闲',
    'error': '错误',
    'queued_tasks': '排队任务',
    'status_distribution': '代理状态分布',
    'status': '状态',
    'uptime': '运行时间',
    'recent_tasks': '最近任务',
    'no_recent_tasks': '暂无最近任务',
    'task_statistics': '任务统计',
    'completed': '已完成',
    'failed': '失败',
    'success_rate': '成功率',
    'avg_duration': '平均耗时',
    'activity_timeline': '代理活动时间线',
    'time': '时间',
    'agent': '代理',
    'task': '任务',
    'task_distribution': '代理任务分布',
}


class AgentMonitor:
    """
    AI代理监控仪表盘组件
    """
    
    def __init__(self, api_base: str = "http://localhost:8000"):
        self.api_base = api_base
    
    def render(self, agents: Dict[str, Any]) -> None:
        """
        渲染代理监控面板
        
        Args:
            agents: 代理实例或状态数据字典
        """
        st.header(LABELS['agent_monitor'])
        
        # 代理概览
        self._render_overview(agents)
        
        # 各代理卡片
        st.subheader(LABELS['agent_details'])
        
        # 为每个代理创建标签页
        agent_names = list(agents.keys()) if agents else []
        
        if agent_names:
            tabs = st.tabs(agent_names)
            
            for i, (name, agent_data) in enumerate(agents.items()):
                with tabs[i]:
                    self._render_agent_card(name, agent_data)
        else:
            st.info(LABELS['no_agents'])
    
    def _render_overview(self, agents: Dict[str, Any]) -> None:
        """渲染代理概览指标"""
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
            st.metric(LABELS['total_agents'], total_agents)
        
        with col2:
            st.metric(LABELS['running'], running_agents)
        
        with col3:
            st.metric(LABELS['idle'], total_agents - running_agents)
        
        with col4:
            st.metric(LABELS['queued_tasks'], total_tasks)
        
        # 代理状态可视化
        self._render_status_chart(agents)
    
    def _render_status_chart(self, agents: Dict[str, Any]) -> None:
        """渲染代理状态条形图"""
        if not agents:
            return
        
        status_counts = {
            LABELS['running']: 0,
            LABELS['idle']: 0,
            LABELS['error']: 0
        }
        
        for agent in agents.values():
            status = self._get_agent_status(agent)
            if status == 'running':
                status_counts[LABELS['running']] += 1
            elif status == 'error':
                status_counts[LABELS['error']] += 1
            else:
                status_counts[LABELS['idle']] += 1
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(status_counts.keys()),
                y=list(status_counts.values()),
                marker_color=['#4CAF50', '#2196F3', '#f44336'],
                text=list(status_counts.values()),
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=LABELS['status_distribution'],
            template='plotly_dark',
            height=200,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_agent_card(self, name: str, agent_data: Any) -> None:
        """渲染单个代理卡片"""
        # 获取状态信息
        if isinstance(agent_data, dict):
            status_info = agent_data
        else:
            status_info = self._extract_agent_info(agent_data)
        
        # 状态徽章
        status = self._get_agent_status(agent_data)
        status_emoji = {
            'running': '🟢',
            '运行中': '🟢',
            'idle': '🟡',
            '空闲': '🟡',
            'error': '🔴',
            '错误': '🔴',
            'stopped': '⚪',
            'stopped': '⚪'
        }.get(status, '⚪')
        
        st.markdown(f"### {status_emoji} {name}")
        
        # 代理指标
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                LABELS['status'],
                status.upper() if status.islower() else status
            )
        
        with col2:
            st.metric(
                LABELS['queued_tasks'],
                status_info.get('queued_tasks', 0)
            )
        
        with col3:
            st.metric(
                LABELS['uptime'],
                status_info.get('uptime', 'N/A')
            )
        
        # 最近任务
        st.subheader(LABELS['recent_tasks'])
        recent_tasks = status_info.get('recent_tasks', [])
        
        if recent_tasks:
            df = pd.DataFrame(recent_tasks)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info(LABELS['no_recent_tasks'])
        
        # 任务统计
        if 'task_stats' in status_info:
            self._render_task_stats(status_info['task_stats'])
    
    def _render_task_stats(self, stats: Dict[str, Any]) -> None:
        """渲染任务统计"""
        st.subheader(LABELS['task_statistics'])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(LABELS['completed'], stats.get('completed', 0))
        
        with col2:
            st.metric(LABELS['failed'], stats.get('failed', 0))
        
        with col3:
            st.metric(LABELS['success_rate'], f"{stats.get('success_rate', 0):.1f}%")
        
        with col4:
            st.metric(LABELS['avg_duration'], f"{stats.get('avg_duration', 0):.2f}秒")
    
    def _get_agent_status(self, agent: Any) -> str:
        """获取代理状态字符串"""
        if isinstance(agent, dict):
            return agent.get('status', 'unknown')
        
        if hasattr(agent, 'status'):
            return agent.status.value if hasattr(agent.status, 'value') else str(agent.status)
        
        if hasattr(agent, 'is_running'):
            return 'running' if agent.is_running else 'stopped'
        
        return 'unknown'
    
    def _extract_agent_info(self, agent: Any) -> Dict[str, Any]:
        """从对象提取代理信息"""
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
    渲染代理活动时间线
    
    Args:
        activities: 活动记录列表
    
    Returns:
        Plotly Figure
    """
    if not activities:
        return go.Figure()
    
    df = pd.DataFrame(activities)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig = go.Figure()
    
    # 为每个代理添加散点
    agents = df['agent'].unique() if 'agent' in df.columns else ['代理']
    
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
                    f"{LABELS['task']}: %{{text}}<br>"
                    "<extra></extra>"
                ),
                text=agent_df['task'] if 'task' in agent_df.columns else ['N/A'] * len(agent_df)
            )
        )
    
    fig.update_layout(
        title=LABELS['activity_timeline'],
        template='plotly_dark',
        height=300,
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(agents))),
            ticktext=list(agents)
        ),
        xaxis_title=LABELS['time'],
        yaxis_title=LABELS['agent']
    )
    
    return fig


def render_agent_task_distribution(
    tasks: List[Dict[str, Any]]
) -> go.Figure:
    """
    渲染任务分布饼图
    
    Args:
        tasks: 任务记录列表
    
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
                textposition='inside',
                textfont=dict(size=12)
            )
        ]
    )
    
    fig.update_layout(
        title=LABELS['task_distribution'],
        template='plotly_dark',
        height=300
    )
    
    return fig


# 导出
__all__ = ['AgentMonitor', 'render_agent_activity_timeline', 'render_agent_task_distribution']
