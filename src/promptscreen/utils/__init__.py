"""Utility functions and classes."""

from .text_preprocessor import TextPreProcessor

# QueryAgent requires langchain-ollama, which is only installed via the
# `eval`/`all` extras. Importing it must not break a core-only install,
# since core guards (e.g. JailbreakInferenceAPI) depend on this package
# for TextPreProcessor.
try:
    from .query_agent import QueryAgent

    _has_query_agent = True
except ImportError:
    QueryAgent = None  # type: ignore
    _has_query_agent = False

__all__ = ["QueryAgent", "TextPreProcessor"]
