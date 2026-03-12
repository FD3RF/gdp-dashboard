"""
安全工具模块
============

提供安全相关的工具函数和装饰器。
"""

import os
import re
import json
import secrets
import logging
import hashlib
from typing import Any, Dict, List, Optional, Callable
from functools import wraps


class SecurityConfig:
    """安全配置"""
    
    # 敏感字段列表（用于脱敏）
    SENSITIVE_FIELDS = [
        'password', 'secret', 'token', 'api_key', 'api_secret',
        'private_key', 'credential', 'auth', 'session'
    ]
    
    # 允许的文件扩展名
    ALLOWED_EXTENSIONS = {'.py', '.json', '.yaml', '.yml', '.txt', '.md', '.csv'}
    
    # 最大文件大小 (10MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024


def safe_json_parse(text: str, default: Any = None) -> Any:
    """
    安全地解析 JSON 字符串
    
    Args:
        text: 可能包含 JSON 的文本
        default: 解析失败时的默认返回值
        
    Returns:
        解析后的对象或默认值
    """
    if not text:
        return default
    
    try:
        # 尝试直接解析
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    try:
        # 尝试提取 JSON 对象
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
        
        # 尝试提取 JSON 数组
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
    except json.JSONDecodeError:
        pass
    
    return default


def sanitize_for_log(data: Any, max_length: int = 100) -> str:
    """
    对日志输出进行脱敏处理
    
    Args:
        data: 要脱敏的数据
        max_length: 最大长度
        
    Returns:
        脱敏后的字符串
    """
    if data is None:
        return "None"
    
    text = str(data)
    
    # 检查是否包含敏感信息
    text_lower = text.lower()
    for field in SecurityConfig.SENSITIVE_FIELDS:
        if field in text_lower:
            # 脱敏处理
            if len(text) > 8:
                return text[:2] + '*' * (len(text) - 4) + text[-2:]
            return '****'
    
    # 截断过长内容
    if len(text) > max_length:
        return text[:max_length] + '...'
    
    return text


def generate_secure_token(length: int = 32) -> str:
    """
    生成安全的随机令牌
    
    Args:
        length: 令牌长度（字节数）
        
    Returns:
        十六进制令牌字符串
    """
    return secrets.token_hex(length)


def hash_password(password: str, salt: str = None) -> tuple:
    """
    安全地哈希密码
    
    Args:
        password: 明文密码
        salt: 盐值（可选）
        
    Returns:
        (哈希值, 盐值)
    """
    if salt is None:
        salt = secrets.token_hex(16)
    
    hashed = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000
    )
    
    return hashed.hex(), salt


def validate_file_path(path: str, base_dir: str = None) -> bool:
    """
    验证文件路径是否安全
    
    Args:
        path: 要验证的路径
        base_dir: 允许的基础目录
        
    Returns:
        路径是否安全
    """
    try:
        path = os.path.abspath(path)
        
        # 检查路径遍历
        if '..' in path:
            return False
        
        # 检查基础目录
        if base_dir:
            base_dir = os.path.abspath(base_dir)
            if not path.startswith(base_dir):
                return False
        
        # 检查扩展名
        ext = os.path.splitext(path)[1].lower()
        if ext not in SecurityConfig.ALLOWED_EXTENSIONS:
            return False
        
        return True
    except Exception:
        return False


def validate_symbol(symbol: str) -> bool:
    """
    验证交易对符号是否有效
    
    Args:
        symbol: 交易对符号
        
    Returns:
        符号是否有效
    """
    if not symbol:
        return False
    
    # 格式: BASE/QUOTE 或 BASE-QUOTE
    pattern = r'^[A-Z]{2,10}[/\-][A-Z]{2,10}$'
    return bool(re.match(pattern, symbol.upper()))


def secure_compare(a: str, b: str) -> bool:
    """
    安全地比较两个字符串（防止时序攻击）
    
    Args:
        a: 第一个字符串
        b: 第二个字符串
        
    Returns:
        是否相等
    """
    return secrets.compare_digest(a, b)


def rate_limit_key(identifier: str, action: str) -> str:
    """
    生成速率限制键
    
    Args:
        identifier: 标识符（如 IP、用户 ID）
        action: 操作类型
        
    Returns:
        速率限制键
    """
    return f"ratelimit:{action}:{identifier}"


def mask_api_key(api_key: str) -> str:
    """
    脱敏 API 密钥用于显示
    
    Args:
        api_key: API 密钥
        
    Returns:
        脱敏后的密钥
    """
    if not api_key or len(api_key) < 8:
        return '****'
    
    return api_key[:4] + '*' * (len(api_key) - 8) + api_key[-4:]


def validate_json_schema(data: Dict, required_fields: List[str]) -> List[str]:
    """
    验证 JSON 数据是否包含必需字段
    
    Args:
        data: 要验证的数据
        required_fields: 必需字段列表
        
    Returns:
        缺失的字段列表
    """
    missing = []
    for field in required_fields:
        if field not in data:
            missing.append(field)
    return missing


def safe_getenv(key: str, default: str = None, required: bool = False) -> Optional[str]:
    """
    安全地获取环境变量
    
    Args:
        key: 环境变量名
        default: 默认值
        required: 是否必需
        
    Returns:
        环境变量值
        
    Raises:
        ValueError: 如果 required=True 但变量不存在
    """
    value = os.environ.get(key, default)
    
    if required and value is None:
        raise ValueError(f"Required environment variable '{key}' is not set")
    
    return value


# 安全异常处理装饰器
def safe_execute(logger: logging.Logger = None, default_return: Any = None):
    """
    安全执行装饰器，避免裸 except
    
    Args:
        logger: 日志记录器
        default_return: 异常时的默认返回值
        
    Returns:
        装饰器
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if logger:
                    logger.error(f"Error in {func.__name__}: {type(e).__name__}: {e}")
                return default_return
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if logger:
                    logger.error(f"Error in {func.__name__}: {type(e).__name__}: {e}")
                return default_return
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# 导出
__all__ = [
    'SecurityConfig',
    'safe_json_parse',
    'sanitize_for_log',
    'generate_secure_token',
    'hash_password',
    'validate_file_path',
    'validate_symbol',
    'secure_compare',
    'rate_limit_key',
    'mask_api_key',
    'validate_json_schema',
    'safe_getenv',
    'safe_execute'
]
