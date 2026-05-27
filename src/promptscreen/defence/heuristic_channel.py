# https://arxiv.org/pdf/2506.06384
# Potential false positives, uncertain on how to correct
import re

from typing_extensions import override

from .abstract_defence import AbstractDefence
from .ds.analysis_result import AnalysisResult


class HeuristicVectorAnalyzer(AbstractDefence):
    def __init__(self, threshold: int, pm_shot_lim: int):
        self.threshold: int = threshold
        self.pm_shot_lim: int = pm_shot_lim
        self.attack_categories: dict[str, list[str]] = {
            "is_ignore": [
                "ignore",
                "reveal",
                "disregard",
                "forget",
                "overlook",
                "regardless",
            ],
            "is_urgent": [
                "urgent",
                "immediate",
                "asap",
                "emergency",
                "critical",
                "time",
            ],
            "is_incentive": [
                "excellent",
                "fantastic",
                "nice",
                "awesome",
                "brilliant",
                "great",
            ],
            "is_covert": [
                "secret",
                "hidden",
                "covert",
                "quiet",
                "silent",
                "discreetly",
                "stealth",
                "sneak",
                "confidential",
                "personal",
            ],
            "is_format_manipulation": [
                "encode",
                "disguising",
                "morse",
                "binary",
                "hexadecimal",
            ],
            "is_hypothetical": [
                "assume",
                "imagine",
                "act",
                "role",
                "play",
                "hypothetical",
                "fictional",
                "scenario",
            ],
            "is_systemic": ["developer", "boss", "manager", "administrator", "creator"],
            "is_immoral": [
                "hitting",
                "amoral",
                "immoral",
                "deceit",
                "irresponsible",
                "offensive",
                "violent",
                "unethical",
                "smack",
                "fake",
                "illegal",
                "biased",
            ],
        }

    def _keyword_bit_vector(self, prompt: str) -> dict[str, int]:
        # Strip punctuation so "urgent!" and "ignore," still match their keywords
        prompt_lower = prompt.lower()
        prompt_lower = prompt_lower.translate(
            str.maketrans("", "", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
        )
        prompt_wds = set(prompt_lower.split())
        flags: dict[str, int] = {}
        for cat, kwds in self.attack_categories.items():
            flags[cat] = 1 if any(kwd in prompt_wds for kwd in kwds) else 0

        return flags

    def _pattern_match_vector(self, prompt: str) -> dict[str, int]:
        flags: dict[str, int] = {}

        qa_pairs = re.findall(
            r"(?:Q:|Question:).*?(?:A:|Answer:)", prompt, re.IGNORECASE | re.DOTALL
        )
        flags["is_shot_attack"] = 1 if len(qa_pairs) > self.pm_shot_lim else 0

        repeated_wd = re.search(r"\b(\w+)\b(?:\s+\1\b){2,}", prompt, re.IGNORECASE)
        flags["is_repeated_token"] = 1 if repeated_wd else 0

        return flags

    def _combine_flags(self, prompt: str) -> dict[str, int]:
        pattern_flg: dict[str, int] = self._pattern_match_vector(prompt)
        keyword_flg: dict[str, int] = self._keyword_bit_vector(prompt)

        return {**pattern_flg, **keyword_flg}

    @override
    def analyse(self, query: str) -> AnalysisResult:
        vec: dict[str, int] = self._combine_flags(query)
        return AnalysisResult("Heuristic channel", sum(vec.values()) < self.threshold)
