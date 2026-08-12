"""Length/complexity features for the SVM jailbreak classifier's FeatureUnion.

This is the single source of truth for ``length_complexity_features``.
``defence/train/train_classifier.py`` and ``defence/linear_svm.py`` both
re-export the name from here rather than defining it themselves, because
``model_artifacts/feature_union.joblib`` pickles the FunctionTransformer
*by module reference* -- moving the function without leaving a name
binding at its old location would break loading of any artifact (including
third-party users' own) trained under the old layout.
"""

import numpy as np


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
