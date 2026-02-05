"""
LLM Adapters Package

Provider-specific adapters for different LLM backends.
Each adapter implements the LLMAdapter interface.
"""

from .ollama import OllamaAdapter
from .vllm import VLLMAdapter
from .tgi import TGIAdapter
from .openai_compatible import OpenAICompatibleAdapter
from .anthropic import AnthropicAdapter
from .openai import OpenAIAdapter
from .google import GoogleAdapter

__all__ = [
    "OllamaAdapter",
    "VLLMAdapter",
    "TGIAdapter",
    "OpenAICompatibleAdapter",
    "AnthropicAdapter",
    "OpenAIAdapter",
    "GoogleAdapter",
]
