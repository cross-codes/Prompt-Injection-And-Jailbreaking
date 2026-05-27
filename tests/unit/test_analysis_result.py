"""Tests for AnalysisResult dataclass."""

from promptscreen.defence.ds.analysis_result import AnalysisResult


class TestAnalysisResult:
    """Test suite for AnalysisResult."""

    def test_safe_result_creation(self):
        """Test creating a safe result."""
        result = AnalysisResult(reason="Clean", is_safe=True)

        assert result.get_verdict() is True
        assert result.get_type() == "Clean"
        assert result.reason == "Clean"
        assert result.is_safe is True
        assert result.confidence is None

    def test_unsafe_result_creation(self):
        """Test creating an unsafe result."""
        result = AnalysisResult(reason="Injection detected", is_safe=False)

        assert result.get_verdict() is False
        assert result.get_type() == "Injection detected"
        assert result.is_safe is False

    def test_result_with_detailed_type(self):
        """Test result with detailed reasoning."""
        result = AnalysisResult(
            reason="Attack pattern detected: keyword 'ignore' found", is_safe=False
        )

        assert not result.get_verdict()
        assert "ignore" in result.get_type().lower()

    def test_result_with_confidence(self):
        """Test that confidence is stored and accessible."""
        result = AnalysisResult(reason="Threat found", is_safe=False, confidence=0.9)

        assert result.confidence == 0.9

    def test_result_without_confidence_defaults_to_none(self):
        """Test that confidence defaults to None when not provided."""
        result = AnalysisResult(reason="Clean", is_safe=True)

        assert result.confidence is None
