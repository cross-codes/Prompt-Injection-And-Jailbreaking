import logging
from pathlib import Path

import joblib
import numpy as np
from typing_extensions import override

from ..utils.text_preprocessor import TextPreProcessor
from .abstract_defence import AbstractDefence
from .ds.analysis_result import AnalysisResult

logger = logging.getLogger(__name__)


def length_complexity_features(texts: list[str]) -> np.ndarray:
    features = []
    attack_keywords = {
        "ignore",
        "system",
        "prompt",
        "act",
        "as",
        "instruction",
        "follow",
        "previous",
    }

    for text in texts:
        char_len = len(text)
        word_len = len(text.split())
        char_no_space = len(text.replace(" ", ""))

        words = text.split()
        if word_len > 0:
            avg_word_len = np.mean([len(w) for w in words])
            punct_ratio = text.count(".") / char_len if char_len > 0 else 0
            attack_density = sum(1 for w in words if w in attack_keywords) / word_len
            repetition_score = (
                max([words.count(w) for w in set(words)]) / word_len
                if word_len > 0
                else 0
            )
        else:
            avg_word_len = 0
            punct_ratio = 0
            attack_density = 0
            repetition_score = 0

        features.append(
            [
                char_len / 1000,
                word_len / 100,
                char_no_space / 1000,
                avg_word_len,
                punct_ratio,
                attack_density,
                repetition_score,
                1.0 / (1 + word_len),
            ]
        )

    return np.array(features)


class JailbreakInferenceAPI(AbstractDefence):
    def __init__(self, model_dir: str):
        model_path = Path(model_dir) / "linear_svm_model.joblib"
        feature_union_path = Path(model_dir) / "feature_union.joblib"

        if not model_path.exists() or not feature_union_path.exists():
            raise FileNotFoundError(
                f"Model or feature_union not found in '{model_dir}'. Please run the enhanced training script first."
            )

        self.model = joblib.load(model_path)
        self.feature_union = joblib.load(feature_union_path)
        self.preprocessor = TextPreProcessor()

    @override
    def analyse(self, query: str) -> AnalysisResult:
        clean_prompt = self.preprocessor.preprocess(query)
        features = self.feature_union.transform([clean_prompt])
        prediction = self.model.predict(features)
        return AnalysisResult("Semantic SVM classifier", prediction[0] != "jailbreak")
