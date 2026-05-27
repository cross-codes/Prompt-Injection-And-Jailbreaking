"""Prompt length guard — blocks prompts that exceed a hard character limit
and flags prompts above a configurable soft-warn threshold.

Long prompts are a common obfuscation vector: many-shot attacks, padding
attacks, and context-overflow exploits all rely on extremely large inputs.
"""

import logging

from typing_extensions import override

from .abstract_defence import AbstractDefence
from .ds.analysis_result import AnalysisResult

logger = logging.getLogger(__name__)

# Rough chars-per-token ratio for most tokenisers (conservative)
_CHARS_PER_TOKEN = 4


class PromptLengthGuard(AbstractDefence):
    """Guard that enforces character-count limits on incoming prompts.

    Parameters
    ----------
    max_chars:
        Hard limit.  Prompts longer than this are blocked (``is_safe=False``).
        Default: 10 000 chars (~2 500 tokens).
    warn_chars:
        Soft limit.  Prompts longer than this but shorter than *max_chars*
        are allowed but returned with ``confidence=0.4`` so downstream chain
        logic can decide whether to act on the warning.
        Default: 4 000 chars (~1 000 tokens).
    """

    def __init__(self, max_chars: int = 10_000, warn_chars: int = 4_000) -> None:
        if warn_chars >= max_chars:
            raise ValueError(
                f"warn_chars ({warn_chars}) must be less than max_chars ({max_chars})"
            )
        self.max_chars = max_chars
        self.warn_chars = warn_chars

    @override
    def analyse(self, query: str) -> AnalysisResult:
        length = len(query)
        approx_tokens = length // _CHARS_PER_TOKEN

        if length > self.max_chars:
            logger.warning(
                "Prompt length %d chars (~%d tokens) exceeds hard limit %d",
                length,
                approx_tokens,
                self.max_chars,
            )
            return AnalysisResult(
                reason=(
                    f"Prompt length {length} chars (~{approx_tokens} tokens) exceeds "
                    f"hard limit of {self.max_chars} chars. "
                    "Unusually long prompts are a common obfuscation vector."
                ),
                is_safe=False,
                confidence=1.0,
            )

        if length > self.warn_chars:
            logger.info(
                "Prompt length %d chars (~%d tokens) exceeds soft warn threshold %d",
                length,
                approx_tokens,
                self.warn_chars,
            )
            return AnalysisResult(
                reason=(
                    f"Prompt length {length} chars (~{approx_tokens} tokens) exceeds "
                    f"soft warning threshold of {self.warn_chars} chars."
                ),
                is_safe=True,
                confidence=0.4,
            )

        return AnalysisResult(
            reason=f"Prompt length {length} chars is within acceptable limits.",
            is_safe=True,
            confidence=1.0,
        )
