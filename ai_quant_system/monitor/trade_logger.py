"""
Trade Logger for trade history management.
"""

import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from core.base import BaseModule


class TradeLogger(BaseModule):
    """
    Logs and manages trade history.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('trade_logger', config)
        
        self._log_dir = Path(self.config.get('log_dir', 'logs/trades'))
        self._trades: List[Dict] = []
        self._max_memory_trades = self.config.get('max_memory_trades', 10000)
        self._flush_interval = self.config.get('flush_interval', 100)
    
    async def initialize(self) -> bool:
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        self._running = True
        return True
    
    async def stop(self) -> bool:
        # Flush remaining trades
        await self._flush_to_file()
        self._running = False
        return True
    
    async def log_trade(self, trade: Dict[str, Any]) -> None:
        """Log a trade."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            **trade
        }
        
        self._trades.append(entry)
        
        # Flush if needed
        if len(self._trades) >= self._flush_interval:
            await self._flush_to_file()
        
        # Trim memory
        if len(self._trades) > self._max_memory_trades:
            self._trades = self._trades[-self._max_memory_trades:]
    
    async def _flush_to_file(self) -> None:
        """Flush trades to file."""
        if not self._trades:
            return
        
        filename = f"trades_{datetime.now().strftime('%Y%m%d')}.jsonl"
        filepath = self._log_dir / filename
        
        with open(filepath, 'a') as f:
            for trade in self._trades[-self._flush_interval:]:
                f.write(json.dumps(trade) + '\n')
    
    def get_trades(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get filtered trades."""
        filtered = self._trades
        
        if start_time:
            filtered = [t for t in filtered if datetime.fromisoformat(t['timestamp']) >= start_time]
        
        if end_time:
            filtered = [t for t in filtered if datetime.fromisoformat(t['timestamp']) <= end_time]
        
        if symbol:
            filtered = [t for t in filtered if t.get('symbol') == symbol]
        
        if strategy:
            filtered = [t for t in filtered if t.get('strategy') == strategy]
        
        return filtered[-limit:]
    
    def get_trade_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get trade statistics."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [t for t in self._trades if datetime.fromisoformat(t['timestamp']) >= cutoff]
        
        if not recent:
            return {'total_trades': 0}
        
        df = pd.DataFrame(recent)
        
        stats = {
            'total_trades': len(recent),
            'buy_trades': len(df[df.get('side', '') == 'buy']) if 'side' in df.columns else 0,
            'sell_trades': len(df[df.get('side', '') == 'sell']) if 'side' in df.columns else 0,
            'total_volume': df.get('value', pd.Series([0])).sum() if 'value' in df.columns else 0,
            'total_fees': df.get('fee', pd.Series([0])).sum() if 'fee' in df.columns else 0,
        }
        
        if 'pnl' in df.columns:
            pnl_data = df['pnl'].dropna()
            stats.update({
                'total_pnl': pnl_data.sum(),
                'winning_trades': (pnl_data > 0).sum(),
                'losing_trades': (pnl_data < 0).sum(),
                'win_rate': (pnl_data > 0).sum() / len(pnl_data) if len(pnl_data) > 0 else 0
            })
        
        return stats
    
    def export_to_csv(self, filepath: str) -> bool:
        """Export trades to CSV."""
        try:
            df = pd.DataFrame(self._trades)
            df.to_csv(filepath, index=False)
            return True
        except Exception as e:
            self.logger.error(f"Export error: {e}")
            return False
    
    def get_daily_summary(self) -> Dict[str, Dict]:
        """Get daily trade summary."""
        df = pd.DataFrame(self._trades)
        
        if df.empty:
            return {}
        
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        
        daily = df.groupby('date').agg({
            'timestamp': 'count',
            'value': 'sum' if 'value' in df.columns else 'count',
            'fee': 'sum' if 'fee' in df.columns else 'count',
            'pnl': 'sum' if 'pnl' in df.columns else 'count'
        }).to_dict('index')
        
        return {str(k): v for k, v in daily.items()}
