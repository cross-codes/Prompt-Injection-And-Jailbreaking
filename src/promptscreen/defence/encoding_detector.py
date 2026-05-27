"""Encoding / steganography detector guard.

Attackers encode malicious instructions to bypass keyword and regex guards:

- **Base64** — ``aWdub3JlIGFsbCBpbnN0cnVjdGlvbnM=`` decodes to
  ``ignore all instructions``
- **Hex dumps** — ``69 67 6e 6f 72 65`` (space-separated) or ``696e6f7265``
- **ROT13** — ``vtaber nyy vafgehpgvbaf`` is ``ignore all instructions``
- **URL / percent encoding** — ``%69%67%6e%6f%72%65`` for control characters
  or whole payloads

The guard decodes detected blobs and checks whether the decoded content
contains common injection keywords, returning a high-confidence block when it
does and a medium-confidence block when a blob is found but the decoded content
looks benign (encoding itself in a prompt context is suspicious).
"""

import base64
import binascii
import logging
import re
import urllib.parse

from typing_extensions import override

from .abstract_defence import AbstractDefence
from .ds.analysis_result import AnalysisResult

logger = logging.getLogger(__name__)

# Minimum blob length to bother decoding (shorter = too many false positives)
_MIN_B64_LEN = 20
_MIN_HEX_LEN = 16  # 8 bytes

# Base64 blob pattern — must end with optional padding, no whitespace inside
_B64_RE = re.compile(r"(?<!\w)([A-Za-z0-9+/]{" + str(_MIN_B64_LEN) + r",}={0,2})(?!\w)")

# Hex dump patterns: spaced (``69 67 6e 6f``) or compact (``696e6f72``)
_HEX_SPACED_RE = re.compile(
    r"(?:(?:[0-9a-fA-F]{2})\s){" + str(_MIN_HEX_LEN // 2) + r",}[0-9a-fA-F]{2}"
)
_HEX_COMPACT_RE = re.compile(r"(?<!\w)([0-9a-fA-F]{" + str(_MIN_HEX_LEN) + r",})(?!\w)")

# URL percent-encoding — at least 4 consecutive encoded bytes
_PCT_ENC_RE = re.compile(r"(?:%[0-9a-fA-F]{2}){4,}")

# Common injection keywords to check inside decoded blobs
_INJECTION_KEYWORDS = frozenset(
    {
        "ignore",
        "disregard",
        "forget",
        "override",
        "bypass",
        "jailbreak",
        "instructions",
        "system",
        "prompt",
        "pretend",
        "roleplay",
        "dan",
        "unrestricted",
        "unlimited",
        "no restrictions",
        "act as",
    }
)

_ROT13_TABLE = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    "NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm",
)


def _contains_injection(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in _INJECTION_KEYWORDS)


def _try_decode_b64(blob: str) -> str | None:
    """Return decoded UTF-8 string or None on failure."""
    try:
        decoded = base64.b64decode(blob + "==").decode("utf-8", errors="replace")
        # Reject if the decoded text is mostly non-printable (binary data)
        printable = sum(1 for c in decoded if c.isprintable() or c in "\n\r\t")
        if printable / max(len(decoded), 1) > 0.7:
            return decoded
    except (binascii.Error, ValueError):
        pass
    return None


def _try_decode_hex(blob: str) -> str | None:
    clean = blob.replace(" ", "")
    if len(clean) % 2 != 0:
        clean = clean[:-1]
    try:
        return bytes.fromhex(clean).decode("utf-8", errors="replace")
    except ValueError:
        return None


def _rot13(text: str) -> str:
    return text.translate(_ROT13_TABLE)


class EncodingDetectorGuard(AbstractDefence):
    """Guard that detects encoded/obfuscated payloads in prompts.

    Parameters
    ----------
    block_on_encoding_alone:
        If ``True``, block any prompt that contains a decodable blob regardless
        of whether the decoded content contains injection keywords.  If
        ``False`` (default), only block when decoded content contains keywords;
        otherwise return ``is_safe=True`` with reduced confidence.
    """

    def __init__(self, block_on_encoding_alone: bool = False) -> None:
        self.block_on_encoding_alone = block_on_encoding_alone

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def _check_base64(self, prompt: str) -> AnalysisResult | None:
        for match in _B64_RE.finditer(prompt):
            blob = match.group(1)
            decoded = _try_decode_b64(blob)
            if decoded is None:
                continue
            if _contains_injection(decoded):
                logger.warning("Base64 blob decodes to injection payload")
                return AnalysisResult(
                    reason=(
                        f"Base64-encoded injection payload detected. "
                        f"Decoded content contains injection keywords: "
                        f"'{decoded[:120]}...'"
                        if len(decoded) > 120
                        else f"Base64-encoded injection payload detected. "
                        f"Decoded: '{decoded}'"
                    ),
                    is_safe=False,
                    confidence=0.95,
                )
            if self.block_on_encoding_alone:
                return AnalysisResult(
                    reason=f"Base64-encoded content detected in prompt (blob: '{blob[:40]}...').",
                    is_safe=False,
                    confidence=0.70,
                )
        return None

    def _check_hex(self, prompt: str) -> AnalysisResult | None:
        for pattern in (_HEX_SPACED_RE, _HEX_COMPACT_RE):
            for match in pattern.finditer(prompt):
                blob = match.group(0)
                decoded = _try_decode_hex(blob)
                if decoded is None:
                    continue
                if _contains_injection(decoded):
                    logger.warning("Hex blob decodes to injection payload")
                    return AnalysisResult(
                        reason=(
                            f"Hex-encoded injection payload detected. "
                            f"Decoded content contains injection keywords: '{decoded[:120]}'"
                        ),
                        is_safe=False,
                        confidence=0.90,
                    )
                if self.block_on_encoding_alone:
                    return AnalysisResult(
                        reason="Hex-encoded content detected in prompt.",
                        is_safe=False,
                        confidence=0.65,
                    )
        return None

    def _check_rot13(self, prompt: str) -> AnalysisResult | None:
        decoded = _rot13(prompt)
        if _contains_injection(decoded):
            logger.warning("Prompt decodes via ROT13 to injection keywords")
            return AnalysisResult(
                reason=(
                    "ROT13-encoded injection payload detected. "
                    f"ROT13-decoded content contains injection keywords: '{decoded[:120]}'"
                ),
                is_safe=False,
                confidence=0.85,
            )
        return None

    def _check_percent_encoding(self, prompt: str) -> AnalysisResult | None:
        for match in _PCT_ENC_RE.finditer(prompt):
            blob = match.group(0)
            try:
                decoded = urllib.parse.unquote(blob)
            except Exception as e:
                logger.debug("Failed to decode percent-encoded blob: '%s'", e)
                continue
            if _contains_injection(decoded):
                logger.warning("URL-encoded blob decodes to injection payload")
                return AnalysisResult(
                    reason=(
                        f"URL/percent-encoded injection payload detected. "
                        f"Decoded: '{decoded[:120]}'"
                    ),
                    is_safe=False,
                    confidence=0.90,
                )
            if self.block_on_encoding_alone:
                return AnalysisResult(
                    reason=f"Suspicious percent-encoded sequence detected: '{blob[:40]}'.",
                    is_safe=False,
                    confidence=0.65,
                )
        return None

    # ------------------------------------------------------------------
    # AbstractDefence
    # ------------------------------------------------------------------

    @override
    def analyse(self, query: str) -> AnalysisResult:
        for check in (
            self._check_base64,
            self._check_hex,
            self._check_rot13,
            self._check_percent_encoding,
        ):
            result = check(query)
            if result is not None:
                return result

        return AnalysisResult(
            reason="No encoded/obfuscated payloads detected.",
            is_safe=True,
            confidence=1.0,
        )
