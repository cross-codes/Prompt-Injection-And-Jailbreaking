import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

# Re-exported for pickle back-compat -- see text_features.py's module docstring.
from ...utils.text_features import length_complexity_features  # noqa: F401

# Importing TextPreProcessor also downloads the NLTK corpora it needs (see
# its module for the resource list) -- no separate nltk.download() here.
from ...utils.text_preprocessor import TextPreProcessor

logger = logging.getLogger(__name__)


class JailbreakClassifier:
    def __init__(self, json_file_path: str, model_output_dir: Optional[str] = None):
        self.json_file_path = json_file_path
        self.model_output_dir = Path(model_output_dir) if model_output_dir else None
        self.preprocessor = TextPreProcessor()

        self.feature_union: Optional[FeatureUnion] = None
        self.model: Optional[LinearSVC] = None

    def classify_prompt(self, prompt: str) -> str:
        if self.model is None or self.feature_union is None:
            raise RuntimeError(
                "Model is not trained. Please run the train() method first."
            )

        clean_prompt = self.preprocessor.preprocess(prompt)
        features = self.feature_union.transform([clean_prompt])
        prediction = self.model.predict(features)
        return str(prediction[0])

    def train(self) -> None:
        if not self.json_file_path:
            raise ValueError("json_file_path must be provided for training.")
        with open(self.json_file_path, encoding="utf-8") as f:
            data = json.load(f)

        df = pd.DataFrame(data)

        label_map = {
            "jailbreak": "jailbreak",
            "prompt-injection": "jailbreak",
            "benign": "benign",
        }
        df["classification"] = df["classification"].map(label_map)
        unknown_labels = set(df["classification"].dropna().unique()) - set(
            label_map.values()
        )
        if unknown_labels:
            logger.warning("Unknown labels found in training data: %s", unknown_labels)

        # Drop rows whose label wasn't recognised
        df = df.dropna(subset=["classification"])

        df["clean_prompt"] = df["prompt"].apply(self.preprocessor.preprocess)

        X = df["clean_prompt"].fillna("")
        y = df["classification"]

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=y
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        feature_union = FeatureUnion(
            [
                ("tfidf", TfidfVectorizer(max_features=17000, ngram_range=(1, 2))),
                ("length_features", FunctionTransformer(length_complexity_features)),
            ]
        )
        X_train_features = feature_union.fit_transform(X_train)
        model = LinearSVC(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=42
        )
        model.fit(X_train_features, y_train)

        logger.info("--- Validation ---")
        X_val_features = feature_union.transform(X_val)
        y_pred_val = model.predict(X_val_features)
        logger.info("\n%s", classification_report(y_val, y_pred_val, zero_division=0))

        logger.info("--- Unseen Test ---")
        X_test_features = feature_union.transform(X_test)
        y_pred = model.predict(X_test_features)
        logger.info("\n%s", classification_report(y_test, y_pred))

        # Retrain on the full dataset for the saved artefact
        self.feature_union = FeatureUnion(
            [
                ("tfidf", TfidfVectorizer(max_features=17000, ngram_range=(1, 2))),
                ("length_features", FunctionTransformer(length_complexity_features)),
            ]
        )
        X_full_features = self.feature_union.fit_transform(X)
        self.model = LinearSVC(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=42
        )
        self.model.fit(X_full_features, y)

        if self.model_output_dir:
            self.model_output_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                self.feature_union, self.model_output_dir / "feature_union.joblib"
            )
            joblib.dump(self.model, self.model_output_dir / "linear_svm_model.joblib")
            logger.info("Final model saved to '%s'.", self.model_output_dir)
