"""Unicode normalization for ML text preprocessing.

Several Unicode-based obfuscation techniques (zero-width characters, RTL
overrides, Unicode tag-block smuggling, combining-mark stacking, full-width
character substitution) can defeat naive tokenization: a prompt obfuscated
this way tokenizes into garbage and gets filtered out entirely, silently
handing the classifier a near-empty feature vector instead of the actual
(recoverable) text.

``normalize_for_model`` reverses these tricks so downstream tokenization
sees the real words. This is deliberately separate from ``UnicodeGuard``
(``defence/unicode_guard.py``), which must keep seeing raw codepoints to do
its job of *flagging* obfuscation -- this module's job is to *recover*
the underlying text for content-based classification.

Character sets are built from explicit codepoint integers rather than
string literals containing the raw characters, so this file never embeds
invisible/directional Unicode characters in its own source text.
"""

import unicodedata

# RTL/LTR directional override and embedding codepoints. Shared with
# UnicodeGuard, which imports these same sets to flag their presence.
_DIRECTIONAL_CODEPOINTS = (
    0x200E,  # LEFT-TO-RIGHT MARK
    0x200F,  # RIGHT-TO-LEFT MARK
    0x202A,  # LEFT-TO-RIGHT EMBEDDING
    0x202B,  # RIGHT-TO-LEFT EMBEDDING
    0x202C,  # POP DIRECTIONAL FORMATTING
    0x202D,  # LEFT-TO-RIGHT OVERRIDE
    0x202E,  # RIGHT-TO-LEFT OVERRIDE
    0x2066,  # LEFT-TO-RIGHT ISOLATE
    0x2067,  # RIGHT-TO-LEFT ISOLATE
    0x2068,  # FIRST STRONG ISOLATE
    0x2069,  # POP DIRECTIONAL ISOLATE
)
DIRECTIONAL_CHARS: frozenset[str] = frozenset(chr(cp) for cp in _DIRECTIONAL_CODEPOINTS)

# Zero-width / soft-hyphen characters used to hide text from regex scanners.
_ZERO_WIDTH_CODEPOINTS = (
    0x00AD,  # SOFT HYPHEN
    0x200B,  # ZERO WIDTH SPACE
    0x200C,  # ZERO WIDTH NON-JOINER
    0x200D,  # ZERO WIDTH JOINER
    0xFEFF,  # BYTE ORDER MARK / ZERO WIDTH NO-BREAK SPACE
)
ZERO_WIDTH_CHARS: frozenset[str] = frozenset(chr(cp) for cp in _ZERO_WIDTH_CODEPOINTS)

# Unicode tag block (U+E0000-E007F): originally for defunct language tags,
# now abused to smuggle text invisibly by offsetting ASCII into this range.
# chr(0xE0000 + ord(c)) round-trips back to the original ASCII character c.
_TAG_BLOCK_START = 0xE0000
_TAG_BLOCK_END = 0xE007F


def _decode_tag_chars(text: str) -> str:
    """Reverse Unicode tag-block smuggling by shifting codepoints back to ASCII."""
    return "".join(
        (
            chr(ord(c) - _TAG_BLOCK_START)
            if _TAG_BLOCK_START <= ord(c) <= _TAG_BLOCK_END
            else c
        )
        for c in text
    )


def normalize_for_model(text: str) -> str:
    """Recover the underlying text from common Unicode obfuscation tricks.

    Order matters:
    1. Decode tag-block smuggling first -- tag characters are Unicode
       category Cf, so stripping invisible/format characters before
       decoding them would destroy the smuggled payload instead of
       revealing it.
    2. Strip zero-width and directional-override characters.
    3. NFKC-normalize to fold full-width characters (e.g. fullwidth 'a'
       -> 'a') and to *compose* decomposed accented characters (e.g.
       'a' + combining acute accent -> a-with-acute) so the next step
       doesn't strip legitimate accents.
    4. Strip any remaining combining marks (category Mn) that had no
       composed form -- this catches marks stacked onto a base character
       purely for obfuscation (e.g. underline-accent-mark smuggling)
       without touching accents that NFKC already composed away.
    """
    text = _decode_tag_chars(text)
    text = "".join(
        c for c in text if c not in DIRECTIONAL_CHARS and c not in ZERO_WIDTH_CHARS
    )
    text = unicodedata.normalize("NFKC", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text
