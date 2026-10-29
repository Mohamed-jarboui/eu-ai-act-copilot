"""Provider layer: one interface, Gemini and Groq backends, fallback and usage records."""

from copilot.llm.base import (
    AllProvidersFailedError,
    AuthError,
    BadRequestError,
    Embedding,
    Generation,
    NotFoundError,
    Provider,
    ProviderError,
    RateLimitError,
    ServerError,
    UnsupportedOperationError,
    Usage,
)
from copilot.llm.factory import build_default_llm
from copilot.llm.router import FallbackLLM

__all__ = [
    "AllProvidersFailedError",
    "AuthError",
    "BadRequestError",
    "Embedding",
    "FallbackLLM",
    "Generation",
    "NotFoundError",
    "Provider",
    "ProviderError",
    "RateLimitError",
    "ServerError",
    "UnsupportedOperationError",
    "Usage",
    "build_default_llm",
]
