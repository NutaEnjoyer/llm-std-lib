"""
Middleware sub-package for llm_std_lib.

Provides a composable pipeline that wraps every LLM request/response.
Custom middleware can be plugged in alongside built-in components.
"""

from .base import BaseMiddleware
from .stack import MiddlewareStack

__all__ = ["BaseMiddleware", "MiddlewareStack"]
