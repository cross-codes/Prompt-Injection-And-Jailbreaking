"""Shared guard-initialization helper for the Hydra-driven example scripts.

Used by both run_pipeline.py (LLM judge mode) and run_stats.py (ground
truth mode), which otherwise need the exact same guard wiring from config.
"""

import json

from omegaconf import DictConfig

from promptscreen.defence import (
    ClassifierCluster,
    HeuristicVectorAnalyzer,
    JailbreakInferenceAPI,
    Scanner,
    ShieldGemma2BClassifier,
    VectorDB,
    VectorDBScanner,
)


def initialize_guards(cfg: DictConfig) -> dict:
    """Initialize guards based on config."""
    guards = {}

    if "heuristic" in cfg.active_defences:
        guards["heuristic"] = HeuristicVectorAnalyzer(3, 3)
        print("- Guard 'heuristic' initialized.")

    if "svm" in cfg.active_defences:
        model_dir = cfg.get("model_dir", "model_artifacts")
        try:
            guards["svm"] = JailbreakInferenceAPI(model_dir)
            print("- Guard 'svm' initialized.")
        except FileNotFoundError as e:
            print(f"Warning: Could not load SVM model: {e}")

    if "cluster" in cfg.active_defences:
        guards["cluster"] = ClassifierCluster()
        print("- Guard 'cluster' initialized.")

    if "shieldgemma" in cfg.active_defences:
        token = cfg.get("huggingface_token")
        if token:
            guards["shieldgemma"] = ShieldGemma2BClassifier(token)
            print("- Guard 'shieldgemma' initialized.")
        else:
            print("Warning: ShieldGemma requires huggingface_token")

    if "scanner" in cfg.active_defences:
        guards["scanner"] = Scanner()  # Uses bundled rules
        print("- Guard 'scanner' initialized.")

    if "vectordb" in cfg.active_defences:
        try:
            db_client = VectorDB(
                model=cfg.vectordb.model,
                collection=cfg.vectordb.collection,
                db_dir=cfg.vectordb.db_dir,
                n_results=cfg.vectordb.n_results,
            )
            # Populate with threat data
            with open(cfg.threat_file) as f:
                threat_data = json.load(f)
                known_threats = [entry["prompt"] for entry in threat_data]
                metadatas = [
                    {"category": entry.get("classification", "unknown")}
                    for entry in threat_data
                ]
                if known_threats:
                    db_client.add_texts(texts=known_threats, metadatas=metadatas)
                    print(f"- VectorDB populated with {len(known_threats)} threats.")

            guards["vectordb"] = VectorDBScanner(db_client, cfg.vectordb.threshold)
            print("- Guard 'vectordb' initialized.")
        except Exception as e:
            print(f"Warning: Could not initialize VectorDB: {e}")

    return guards
