from typing import Optional


class AnalysisResult:
    def __init__(self, reason: str, is_safe: bool, confidence: Optional[float] = None):
        self.reason: str = reason
        self.is_safe: bool = is_safe
        # Confidence in the verdict (0.0–1.0). None means the guard does not
        # produce a calibrated confidence score.
        self.confidence: Optional[float] = confidence

    def get_verdict(self) -> bool:
        return self.is_safe

    def get_type(self) -> str:
        """Return the reason string. Kept for backward compatibility."""
        return self.reason
