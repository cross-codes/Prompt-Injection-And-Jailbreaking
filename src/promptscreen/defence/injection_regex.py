# credits to zhu-weije/prompt-guard

import re
import unicodedata
from collections.abc import Iterator

from typing_extensions import override

from .abstract_defence import AbstractDefence
from .ds.analysis_result import AnalysisResult

SENSITIVE_FILES = (
    r"(settings\.json|\.bashrc|\.zshrc|\.profile|config\.toml|credentials)"
)
FILE_WRITE_PATTERN = re.compile(
    rf"(?:>|>>|sed|vim|nano|echo|cat)\s*.*?{SENSITIVE_FILES}", re.IGNORECASE
)

DNS_COMMAND_PATTERN = re.compile(
    r"`\s*(?:nslookup|dig|host|ping).*?`", re.IGNORECASE | re.DOTALL
)

INVISIBLE_CATEGORIES = ("Cf", "Cc", "Zs")
ALLOWED_CHARS = (" ", "\n", "\r")

MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[.*?\]\((https?://.*?)\)")


class Vulnerability:
    def __init__(self, category: str, description: str, confidence: float):
        self.category: str = category
        self.description: str = description
        self.confidence: float = confidence


class InjectionScanner(AbstractDefence):
    def _find_file_write_issues(self, prompt: str) -> Iterator[Vulnerability]:
        matches = FILE_WRITE_PATTERN.finditer(prompt)
        for match in matches:
            snippet = match.group(0).strip()
            filename = match.group(1)
            yield Vulnerability(
                category="Privilege Escalation",
                description=(
                    "An attempt to write to or edit a sensitive configuration file was "
                    f"detected. This could be an attempt to alter system behavior or "
                    f"escalate privileges. Sensitive file '{filename}' targeted in "
                    f"snippet: `{snippet}`"
                ),
                confidence=0.9,
            )

    def _find_dns_issues(self, prompt: str) -> Iterator[Vulnerability]:
        matches = DNS_COMMAND_PATTERN.finditer(prompt)
        for match in matches:
            command_found = match.group(0).strip()
            yield Vulnerability(
                category="Data Exfiltration",
                description=(
                    "A command known for DNS lookups was found with suspicious "
                    "patterns, suggesting data could be exfiltrated via DNS queries. "
                    f"Command snippet found: `{command_found}`"
                ),
                confidence=0.7,
            )

    def _find_invisible_chars(self, prompt: str) -> Iterator[Vulnerability]:
        for i, char in enumerate(prompt):
            if char in ALLOWED_CHARS:
                continue

            category = unicodedata.category(char)
            if category in INVISIBLE_CATEGORIES:
                yield Vulnerability(
                    category="Obfuscation",
                    description=(
                        f"An invisible or non-printable character was found at "
                        f"position {i}. This could be an attempt to hide malicious "
                        f"instructions. Character: '{char}' (U+{ord(char):04X}), "
                        f"Category: {category}."
                    ),
                    confidence=0.9,
                )
                break

    def _find_markdown_exfiltration(self, prompt: str) -> Iterator[Vulnerability]:
        matches = MARKDOWN_IMAGE_PATTERN.finditer(prompt)
        for match in matches:
            url = match.group(1)
            yield Vulnerability(
                category="Data Exfiltration",
                description=(
                    "A Markdown image tag was found with a remote URL. "
                    "This can be used to exfiltrate data if the prompt is rendered "
                    f"by another system or user. URL found: {url}"
                ),
                confidence=0.8,
            )

    @override
    def analyse(self, query: str) -> AnalysisResult:
        all_vulnerabilities = (
            list(self._find_file_write_issues(query))
            + list(self._find_dns_issues(query))
            + list(self._find_invisible_chars(query))
            + list(self._find_markdown_exfiltration(query))
        )

        if not all_vulnerabilities:
            return AnalysisResult(reason="No vulnerabilities found.", is_safe=True)

        descriptions = [v.description for v in all_vulnerabilities]
        problem_string = "\n---\n".join(descriptions)
        avg_confidence = sum(v.confidence for v in all_vulnerabilities) / len(
            all_vulnerabilities
        )
        return AnalysisResult(
            reason=problem_string, is_safe=False, confidence=avg_confidence
        )
