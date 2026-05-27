"""Unicode guard — detects prompt-injection obfuscation via Unicode tricks.

Three attack classes are covered:

1. **RTL overrides** — U+202A–202E, U+2066–2069, U+200E/F reverse the visual
   rendering of text while the model still processes the raw codepoints.
2. **Zero-width characters** — U+200B (ZWSP), U+200C (ZWNJ), U+200D (ZWJ),
   U+FEFF (BOM), U+00AD (soft hyphen) are invisible to humans but present in
   the token stream, making keyword guards blind to obfuscated words.
3. **Mixed-script homograph attacks** — using Cyrillic 'а' instead of Latin 'a'
   (and similar lookalikes from Greek, Armenian, etc.) to fool exact-match
   detectors while the word *looks* identical to a human reader.

Note: the ``InjectionScanner`` already catches generic invisible characters
(Unicode categories Cf/Cc/Zs).  This guard focuses on the three specific
attack classes above.
"""

import logging
import unicodedata

from typing_extensions import override

from .abstract_defence import AbstractDefence
from .ds.analysis_result import AnalysisResult

logger = logging.getLogger(__name__)

# RTL/LTR directional override and embedding codepoints
_RTL_OVERRIDE_CHARS: frozenset[str] = frozenset(
    "\u200e"  # LEFT-TO-RIGHT MARK
    "\u200f"  # RIGHT-TO-LEFT MARK
    "\u202a"  # LEFT-TO-RIGHT EMBEDDING
    "\u202b"  # RIGHT-TO-LEFT EMBEDDING
    "\u202c"  # POP DIRECTIONAL FORMATTING
    "\u202d"  # LEFT-TO-RIGHT OVERRIDE
    "\u202e"  # RIGHT-TO-LEFT OVERRIDE  ← most commonly abused
    "\u2066"  # LEFT-TO-RIGHT ISOLATE
    "\u2067"  # RIGHT-TO-LEFT ISOLATE
    "\u2068"  # FIRST STRONG ISOLATE
    "\u2069"  # POP DIRECTIONAL ISOLATE
)

# Zero-width / soft-hyphen characters used to hide text from regex scanners
_ZERO_WIDTH_CHARS: frozenset[str] = frozenset(
    "\u00ad"  # SOFT HYPHEN
    "\u200b"  # ZERO WIDTH SPACE
    "\u200c"  # ZERO WIDTH NON-JOINER
    "\u200d"  # ZERO WIDTH JOINER
    "\ufeff"  # BYTE ORDER MARK / ZERO WIDTH NO-BREAK SPACE
)

# Scripts that are commonly used for homograph attacks against Latin text
_CONFUSABLE_SCRIPTS = frozenset(
    {
        "CYRILLIC",
        "GREEK",
        "ARMENIAN",
        "GEORGIAN",
        "CHEROKEE",
        "UNIFIED CANADIAN ABORIGINAL SYLLABICS",
    }
)


def _char_script(char: str) -> str:
    """Return the Unicode script name for a character (best-effort)."""
    name = unicodedata.name(char, "")
    # Unicode names start with the script name, e.g. "LATIN SMALL LETTER A"
    return name.split()[0] if name else "UNKNOWN"


class UnicodeGuard(AbstractDefence):
    """Guard that detects Unicode-based obfuscation in prompts.

    Parameters
    ----------
    min_word_len:
        Minimum word length to check for mixed-script homographs.  Short words
        (e.g. "a", "I") are excluded to reduce false positives.
        Default: 4.
    """

    def __init__(self, min_word_len: int = 4) -> None:
        self.min_word_len = min_word_len

    # ------------------------------------------------------------------
    # Internal detectors
    # ------------------------------------------------------------------

    def _find_rtl_overrides(self, prompt: str) -> list[str]:
        return [c for c in prompt if c in _RTL_OVERRIDE_CHARS]

    def _find_zero_width(self, prompt: str) -> list[str]:
        return [c for c in prompt if c in _ZERO_WIDTH_CHARS]

    def _find_mixed_script_words(self, prompt: str) -> list[str]:
        """Return words that mix Latin characters with a confusable script."""
        suspicious: list[str] = []
        for word in prompt.split():
            if len(word) < self.min_word_len:
                continue
            scripts: set[str] = set()
            for ch in word:
                if ch.isalpha():
                    scripts.add(_char_script(ch))
            if "LATIN" in scripts and scripts & _CONFUSABLE_SCRIPTS:
                suspicious.append(word)
        return suspicious

    # ------------------------------------------------------------------
    # AbstractDefence
    # ------------------------------------------------------------------

    @override
    def analyse(self, query: str) -> AnalysisResult:
        rtl = self._find_rtl_overrides(query)
        if rtl:
            chars_hex = ", ".join(f"U+{ord(c):04X}" for c in rtl[:5])
            logger.warning("RTL override chars detected: %s", chars_hex)
            return AnalysisResult(
                reason=(
                    f"RTL/directional override character(s) detected ({chars_hex}). "
                    "These can reverse the visual rendering of text to hide malicious "
                    "instructions from human reviewers."
                ),
                is_safe=False,
                confidence=0.95,
            )

        zw = self._find_zero_width(query)
        if zw:
            chars_hex = ", ".join(f"U+{ord(c):04X}" for c in zw[:5])
            logger.warning("Zero-width chars detected: %s", chars_hex)
            return AnalysisResult(
                reason=(
                    f"Zero-width / soft-hyphen character(s) detected ({chars_hex}). "
                    "These are invisible to humans but present in the token stream "
                    "and can be used to bypass keyword-matching defences."
                ),
                is_safe=False,
                confidence=0.90,
            )

        mixed = self._find_mixed_script_words(query)
        if mixed:
            examples = ", ".join(f"'{w}'" for w in mixed[:3])
            logger.warning("Mixed-script words detected: %s", examples)
            return AnalysisResult(
                reason=(
                    f"Mixed-script homograph word(s) detected: {examples}. "
                    "These words blend Latin characters with visually identical "
                    "characters from another script (e.g. Cyrillic) to fool "
                    "exact-match detectors."
                ),
                is_safe=False,
                confidence=0.85,
            )

        return AnalysisResult(
            reason="No suspicious Unicode patterns detected.",
            is_safe=True,
            confidence=1.0,
        )
