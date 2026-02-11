"""
LLM Adapters Package

vLLM adapter for high-performance inference via the vLLM deployment.
"""

from .vllm import VLLMAdapter

__all__ = [
    "VLLMAdapter",
]
