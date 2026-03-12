# config.py

# 智能体参数
AGENT_CONFIG = {
    "state_dim": 256,      # 状态向量维度 (全息感知层输出)
    "action_dim": 4,       # 动作空间: [开多, 开空, 平仓, 观望]
    "hidden_dim": 512,     # 神经网络隐藏层维度
    "max_position": 0.2    # 最大仓位比例 (凯利公式约束)
}

# 风控参数
RISK_CONFIG = {
    "single_loss_limit": 0.02,   # 单笔最大亏损 2%
    "daily_loss_limit": 0.05,    # 日内最大亏损 5%
    "max_position": 0.2,         # 最大仓位 20%
    "max_leverage": 5,           # 最大杠杆
    "api_timeout": 500,          # API 超时熔断
    "black_swan_threshold": 0.1, # 黑天鹅阈值 10%
}

# 交易对配置
SYMBOLS = ['ETH/USDT', 'BTC/USDT']

# 时间周期
TIMEFRAME = '5m'  # 5分钟K线

# OKX API 配置
EXCHANGE_CONFIG = {
    'name': 'okx',
    'testnet': True,
    'api_key': '',  # 填入你的API Key
    'api_secret': '',  # 填入你的API Secret
    'passphrase': '',  # 填入你的Passphrase
}
