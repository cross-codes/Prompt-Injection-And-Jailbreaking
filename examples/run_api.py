"""Run FastAPI server with all guards.

This starts an API server that allows testing guards via HTTP requests.
"""

import os

import hydra
import uvicorn
from omegaconf import DictConfig

from promptscreen.api import create_app
from promptscreen.defence import (
    ClassifierCluster,
    HeuristicVectorAnalyzer,
    JailbreakInferenceAPI,
    Scanner,
    ShieldGemma2BClassifier,
    VectorDB,
    VectorDBScanner,
)


def initialize_all_guards(cfg: DictConfig) -> dict:
    """Initialize all available guards."""
    guards = {}

    # Always initialize these (no heavy dependencies)
    guards["heuristic"] = HeuristicVectorAnalyzer(3, 3)
    guards["scanner"] = Scanner()

    # SVM (if model exists)
    try:
        guards["svm"] = JailbreakInferenceAPI(cfg.get("model_dir", "model_artifacts"))
    except FileNotFoundError:
        print("Warning: SVM model not found, skipping.")

    # Cluster (requires transformers)
    try:
        guards["cluster"] = ClassifierCluster()
    except Exception as e:
        print(f"Warning: Could not initialize cluster: {e}")

    # ShieldGemma (requires huggingface token)
    if cfg.get("huggingface_token"):
        try:
            guards["shieldgemma"] = ShieldGemma2BClassifier(cfg.huggingface_token)
        except Exception as e:
            print(f"Warning: Could not initialize ShieldGemma: {e}")

    # VectorDB (requires chromadb)
    try:
        db_client = VectorDB(
            model=cfg.vectordb.model,
            collection=cfg.vectordb.collection,
            db_dir=cfg.vectordb.db_dir,
            n_results=cfg.vectordb.n_results,
        )
        # Populate with threat data
        import json

        with open(cfg.threat_file) as f:
            threat_data = json.load(f)
            known_threats = [entry["prompt"] for entry in threat_data]
            metadatas = [
                {"category": entry.get("classification", "unknown")}
                for entry in threat_data
            ]
            if known_threats:
                db_client.add_texts(texts=known_threats, metadatas=metadatas)

        guards["vectordb"] = VectorDBScanner(db_client, cfg.vectordb.threshold)
    except Exception as e:
        print(f"Warning: Could not initialize VectorDB: {e}")

    return guards


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=" * 60)
    print("PROMPTSCREEN - API Server")
    print("=" * 60)

    guards = initialize_all_guards(cfg)
    print(f"\nInitialized {len(guards)} guards: {list(guards.keys())}")

    host = cfg.api.get("host", "127.0.0.1")
    port = cfg.api.get("port", 8000)
    # Set PROMPTSCREEN_API_KEY to require an X-API-Key header on every
    # request. Unset (the default) leaves the API unauthenticated -- fine
    # for localhost-only dev use, not for exposing it beyond that.
    api_key = os.getenv("PROMPTSCREEN_API_KEY")

    print(f"\nStarting API server at http://{host}:{port}")
    print(f"API docs available at: http://{host}:{port}/docs")
    if not api_key and host not in ("127.0.0.1", "localhost"):
        print(
            "WARNING: binding to a non-localhost host with no "
            "PROMPTSCREEN_API_KEY set -- this API will be unauthenticated."
        )
    print("=" * 60)

    app = create_app(guards, api_key=api_key)
    uvicorn.run(app, host=host, port=port)  # noqa: S104


if __name__ == "__main__":
    main()
