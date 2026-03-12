"""
Model Manager for AI model management and inference.
"""

import asyncio
import json
import logging
import aiohttp
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from core.base import BaseModule
from core.exceptions import ModelException, AgentTimeoutException


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    provider: str
    host: str
    port: int
    timeout: int = 300
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class ModelResponse:
    """Model response container."""
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str
    latency: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelManager(BaseModule):
    """
    Manages AI model connections and inference.
    Supports Ollama, OpenAI-compatible APIs.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('model_manager', config)
        self._models: Dict[str, ModelConfig] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._default_model = self.config.get('default_model', 'qwen2.5-coder')
        
        # Add default Ollama model
        self._add_default_model()
    
    def _add_default_model(self) -> None:
        """Add default Ollama model configuration."""
        default_config = ModelConfig(
            name=self.config.get('model', 'qwen2.5-coder:latest'),
            provider='ollama',
            host=self.config.get('host', 'localhost'),
            port=self.config.get('port', 11434),
            timeout=self.config.get('timeout', 300),
            max_tokens=self.config.get('max_tokens', 4096),
            temperature=self.config.get('temperature', 0.7)
        )
        self._models['default'] = default_config
    
    async def initialize(self) -> bool:
        """Initialize the model manager."""
        self.logger.info("Initializing model manager...")
        
        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=300)
        self._session = aiohttp.ClientSession(timeout=timeout)
        
        # Check model availability
        await self._check_model_availability()
        
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        """Start the model manager."""
        self._running = True
        self._start_time = datetime.now()
        return True
    
    async def stop(self) -> bool:
        """Stop the model manager."""
        if self._session:
            await self._session.close()
        self._running = False
        return True
    
    async def _check_model_availability(self) -> bool:
        """Check if the model is available."""
        try:
            model_config = self._models.get('default')
            if not model_config:
                return False
            
            async with self._session.get(f"{model_config.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = [m['name'] for m in data.get('models', [])]
                    self.logger.info(f"Available models: {models}")
                    return True
        except Exception as e:
            self.logger.warning(f"Could not check model availability: {e}")
        return False
    
    def register_model(self, model_id: str, config: ModelConfig) -> None:
        """Register a new model."""
        self._models[model_id] = config
        self.logger.info(f"Registered model: {model_id}")
    
    def get_model_config(self, model_id: str = 'default') -> Optional[ModelConfig]:
        """Get model configuration."""
        return self._models.get(model_id)
    
    async def generate(
        self,
        prompt: str,
        model_id: str = 'default',
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Generate text using the model.
        
        Args:
            prompt: Input prompt
            model_id: Model identifier
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            **kwargs: Additional parameters
        
        Returns:
            ModelResponse object
        """
        model_config = self._models.get(model_id)
        if not model_config:
            raise ModelException(f"Model not found: {model_id}")
        
        start_time = datetime.now()
        
        try:
            response = await self._call_ollama(
                model_config=model_config,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
                **kwargs
            )
            
            latency = (datetime.now() - start_time).total_seconds()
            response.latency = latency
            return response
            
        except asyncio.TimeoutError:
            raise AgentTimeoutException(
                agent_name='ModelManager',
                timeout=model_config.timeout
            )
        except Exception as e:
            raise ModelException(f"Model inference failed: {e}", model_name=model_config.name)
    
    async def _call_ollama(
        self,
        model_config: ModelConfig,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> ModelResponse:
        """Call Ollama API."""
        url = f"{model_config.base_url}/api/generate"
        
        payload = {
            'model': model_config.name,
            'prompt': prompt,
            'stream': False,
            'options': {
                'temperature': temperature or model_config.temperature,
                'num_predict': max_tokens or model_config.max_tokens,
                'top_p': kwargs.get('top_p', model_config.top_p),
                'repeat_penalty': kwargs.get('repeat_penalty', model_config.repeat_penalty),
            }
        }
        
        if system_prompt:
            payload['system'] = system_prompt
        
        if stop:
            payload['options']['stop'] = stop
        
        timeout = aiohttp.ClientTimeout(total=model_config.timeout)
        
        async with self._session.post(url, json=payload, timeout=timeout) as response:
            if response.status != 200:
                error_text = await response.text()
                raise ModelException(f"Ollama API error: {error_text}")
            
            data = await response.json()
            
            return ModelResponse(
                content=data.get('response', ''),
                model=model_config.name,
                prompt_tokens=data.get('prompt_eval_count', 0),
                completion_tokens=data.get('eval_count', 0),
                total_tokens=data.get('prompt_eval_count', 0) + data.get('eval_count', 0),
                finish_reason='stop',
                latency=0.0,
                metadata={'raw_response': data}
            )
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model_id: str = 'default',
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Chat completion using the model.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model_id: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters
        
        Returns:
            ModelResponse object
        """
        model_config = self._models.get(model_id)
        if not model_config:
            raise ModelException(f"Model not found: {model_id}")
        
        # Convert messages to prompt for Ollama
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        prompt = "\n".join(prompt_parts) + "\nAssistant:"
        
        return await self.generate(
            prompt=prompt,
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    async def embed(self, text: str, model_id: str = 'default') -> List[float]:
        """
        Generate embeddings for text.
        
        Args:
            text: Input text
            model_id: Model identifier
        
        Returns:
            Embedding vector
        """
        model_config = self._models.get(model_id)
        if not model_config:
            raise ModelException(f"Model not found: {model_id}")
        
        url = f"{model_config.base_url}/api/embeddings"
        
        payload = {
            'model': model_config.name,
            'prompt': text
        }
        
        async with self._session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise ModelException(f"Embedding API error: {error_text}")
            
            data = await response.json()
            return data.get('embedding', [])
    
    async def pull_model(self, model_name: str, model_id: str = 'default') -> bool:
        """Pull a model from Ollama registry."""
        model_config = self._models.get(model_id)
        if not model_config:
            return False
        
        url = f"{model_config.base_url}/api/pull"
        payload = {'name': model_name}
        
        try:
            async with self._session.post(url, json=payload) as response:
                return response.status == 200
        except Exception as e:
            self.logger.error(f"Failed to pull model: {e}")
            return False
    
    def list_models(self) -> List[str]:
        """List registered models."""
        return list(self._models.keys())
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        model_available = await self._check_model_availability()
        return {
            'healthy': self._initialized and model_available,
            'name': self.name,
            'timestamp': datetime.now().isoformat(),
            'models': self.list_models()
        }
