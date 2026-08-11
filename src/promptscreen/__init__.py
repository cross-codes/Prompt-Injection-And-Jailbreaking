"""PromptScreen - Production-ready prompt injection defense for LLMs.

A comprehensive library for defending Large Language Models against
prompt injection and jailbreak attacks.
"""

__version__ = "0.3.0"

# Core defence classes (always available)
from .defence import (
    AbstractDefence,
    AnalysisResult,
    EncodingDetectorGuard,
    HeuristicVectorAnalyzer,
    InjectionScanner,
    JailbreakInferenceAPI,
    PolymorphicPromptAssembler,
    PromptLengthGuard,
    RateLimitGuard,
    Scanner,
    UnicodeGuard,
    VectorDB,
    VectorDBScanner,
)

# Optional ML guards (require torch/transformers)
try:
    from .defence import ClassifierCluster, ShieldGemma2BClassifier

    _has_ml = True
except (ImportError, AttributeError):
    ClassifierCluster = None  # type: ignore
    ShieldGemma2BClassifier = None  # type: ignore
    _has_ml = False

# Evaluation framework (optional - only if user installs [eval] extras)
try:
    from .evaluation import AttackEvaluator, MetricsCalculator

    _has_evaluation = True
except ImportError:
    _has_evaluation = False
    AttackEvaluator = None  # type: ignore
    MetricsCalculator = None  # type: ignore

# API server (optional - only if user installs [api] extras)
try:
    from .api import create_app

    _has_api = True
except ImportError:
    _has_api = False
    create_app = None  # type: ignore

__all__ = [
    "__version__",
    "AbstractDefence",
    "AnalysisResult",
    "EncodingDetectorGuard",
    "HeuristicVectorAnalyzer",
    "InjectionScanner",
    "JailbreakInferenceAPI",
    "PolymorphicPromptAssembler",
    "PromptLengthGuard",
    "RateLimitGuard",
    "Scanner",
    "UnicodeGuard",
    "VectorDB",
    "VectorDBScanner",
    "ClassifierCluster",
    "ShieldGemma2BClassifier",
    "AttackEvaluator",
    "MetricsCalculator",
    "create_app",
]
