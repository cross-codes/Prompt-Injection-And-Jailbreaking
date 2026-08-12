import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from typing_extensions import override

# Re-exported for pickle back-compat -- see text_features.py's module docstring.
from ..utils.text_features import length_complexity_features  # noqa: F401
from ..utils.text_preprocessor import TextPreProcessor
from .abstract_defence import AbstractDefence
from .ds.analysis_result import AnalysisResult

logger = logging.getLogger(__name__)

# Below this raw length, legitimate short prompts ("ok", "the", "123") can
# fully empty out during normal preprocessing (punctuation/stopword removal)
# with no obfuscation involved -- the empty-text short-circuit below should
# only fire on inputs long enough that vanishing entirely is itself unusual.
_MIN_LEN_FOR_EMPTY_CHECK = 20


class JailbreakInferenceAPI(AbstractDefence):
    """SVM-based jailbreak classifier loaded from a directory of joblib artifacts.

    Security note: ``joblib.load`` deserializes via pickle and will execute
    arbitrary code embedded in a malicious artifact. Only point ``model_dir``
    at artifacts you trust (e.g. the bundled ``model_artifacts/`` or your own
    training output) -- never at a directory populated from an untrusted or
    user-controlled source.
    """

    def __init__(self, model_dir: str):
        model_path = Path(model_dir) / "linear_svm_model.joblib"
        feature_union_path = Path(model_dir) / "feature_union.joblib"

        if not model_path.exists() or not feature_union_path.exists():
            raise FileNotFoundError(
                f"Model or feature_union not found in '{model_dir}'. Please run the enhanced training script first."
            )

        # joblib.load executes arbitrary code on deserialization (it's pickle
        # underneath) -- see class docstring.
        self.model = joblib.load(model_path)
        self.feature_union = joblib.load(feature_union_path)
        self.preprocessor = TextPreProcessor()

    def _decision_confidence(self, features: np.ndarray) -> Optional[float]:
        """Map the SVM's decision margin to a 0.5-1.0 confidence score.

        LinearSVC has no ``predict_proba`` (that requires the heavier ``SVC``
        with ``probability=True``), but ``decision_function`` gives the
        signed distance to the separating hyperplane -- larger magnitude
        means the prompt sits further from the boundary, i.e. the guard is
        more confident in whichever verdict it returned. This is an
        uncalibrated heuristic, not a true probability.
        """
        try:
            margin = float(self.model.decision_function(features)[0])
        except (AttributeError, IndexError, TypeError):
            return None
        return float(1.0 / (1.0 + np.exp(-abs(margin))))

    @override
    def analyse(self, query: str) -> AnalysisResult:
        clean_prompt = self.preprocessor.preprocess(query)

        raw = query.strip()
        if not clean_prompt and len(raw) >= _MIN_LEN_FOR_EMPTY_CHECK:
            # A substantial prompt that normalizes to nothing is itself
            # anomalous (heavy obfuscation, non-alphabetic content, or an
            # obfuscation technique normalize_for_model doesn't cover yet).
            # Flag it rather than silently handing the model a near-zero
            # feature vector it has no real basis to judge.
            return AnalysisResult(
                "Semantic SVM classifier: prompt contained no analysable text "
                "after normalisation",
                False,
                confidence=0.3,
            )

        features = self.feature_union.transform([clean_prompt])
        prediction = self.model.predict(features)
        confidence = self._decision_confidence(features)
        return AnalysisResult(
            "Semantic SVM classifier",
            prediction[0] != "jailbreak",
            confidence=confidence,
        )
