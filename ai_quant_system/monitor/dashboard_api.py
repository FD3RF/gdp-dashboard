"""
Dashboard API for web interface.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from core.base import BaseModule


class DashboardAPI(BaseModule):
    """
    REST API for dashboard access.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('dashboard_api', config)
        
        self._app: Optional[FastAPI] = None
        self._port = self.config.get('port', 8000)
        self._host = self.config.get('host', '0.0.0.0')
        
        self._system: Dict[str, Any] = {}
    
    def set_system_reference(self, system: Dict[str, Any]) -> None:
        """Set reference to main system components."""
        self._system = system
    
    async def initialize(self) -> bool:
        self._app = FastAPI(
            title="AI Quant Trading System",
            description="API for monitoring and controlling the trading system",
            version="1.0.0"
        )
        
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_routes()
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        import asyncio
        
        config = uvicorn.Config(
            app=self._app,
            host=self._host,
            port=self._port,
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        asyncio.create_task(server.serve())
        
        self._running = True
        return True
    
    async def stop(self) -> bool:
        self._running = False
        return True
    
    def _setup_routes(self) -> None:
        """Setup API routes."""
        
        @self._app.get("/")
        async def root():
            return {"message": "AI Quant Trading System API", "version": "1.0.0"}
        
        @self._app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat()
            }
        
        @self._app.get("/system/status")
        async def system_status():
            return self._get_system_status()
        
        @self._app.get("/strategies")
        async def list_strategies():
            return self._get_strategies()
        
        @self._app.get("/strategies/{name}")
        async def get_strategy(name: str):
            return self._get_strategy(name)
        
        @self._app.get("/positions")
        async def get_positions():
            return self._get_positions()
        
        @self._app.get("/risk")
        async def get_risk():
            return self._get_risk_metrics()
        
        @self._app.get("/performance")
        async def get_performance():
            return self._get_performance()
        
        @self._app.get("/alerts")
        async def get_alerts():
            return self._get_alerts()
        
        @self._app.post("/strategies/{name}/start")
        async def start_strategy(name: str):
            return await self._start_strategy(name)
        
        @self._app.post("/strategies/{name}/stop")
        async def stop_strategy(name: str):
            return await self._stop_strategy(name)
    
    def _get_system_status(self) -> Dict:
        """Get overall system status."""
        return {
            "running": self._running,
            "timestamp": datetime.now().isoformat(),
            "components": {
                "agents": len(self._system.get('agents', {})),
                "strategies": len(self._system.get('strategies', {})),
                "data_collectors": len(self._system.get('data_collectors', {}))
            }
        }
    
    def _get_strategies(self) -> List[Dict]:
        """Get all strategies."""
        strategies = self._system.get('strategies', {})
        return [
            {"name": name, "status": "running"}
            for name in strategies
        ]
    
    def _get_strategy(self, name: str) -> Dict:
        """Get strategy details."""
        strategies = self._system.get('strategies', {})
        if name in strategies:
            strategy = strategies[name]
            if hasattr(strategy, 'get_status'):
                return strategy.get_status()
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    def _get_positions(self) -> List[Dict]:
        """Get current positions."""
        return []
    
    def _get_risk_metrics(self) -> Dict:
        """Get risk metrics."""
        risk = self._system.get('risk_dashboard')
        if risk and hasattr(risk, 'get_dashboard'):
            return risk.get_dashboard(100000)
        return {}
    
    def _get_performance(self) -> Dict:
        """Get performance metrics."""
        return {}
    
    def _get_alerts(self) -> List[Dict]:
        """Get recent alerts."""
        alert_system = self._system.get('alert_system')
        if alert_system and hasattr(alert_system, 'get_alerts'):
            return alert_system.get_alerts()
        return []
    
    async def _start_strategy(self, name: str) -> Dict:
        """Start a strategy."""
        strategies = self._system.get('strategies', {})
        if name in strategies:
            strategy = strategies[name]
            if hasattr(strategy, 'start'):
                await strategy.start()
                return {"status": "started", "name": name}
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    async def _stop_strategy(self, name: str) -> Dict:
        """Stop a strategy."""
        strategies = self._system.get('strategies', {})
        if name in strategies:
            strategy = strategies[name]
            if hasattr(strategy, 'stop'):
                await strategy.stop()
                return {"status": "stopped", "name": name}
        raise HTTPException(status_code=404, detail="Strategy not found")
