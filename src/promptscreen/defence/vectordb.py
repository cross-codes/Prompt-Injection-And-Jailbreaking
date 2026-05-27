from typing import Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from pydantic import BaseModel
from typing_extensions import override

from .abstract_defence import AbstractDefence
from .ds.analysis_result import AnalysisResult


class VectorDB:
    def __init__(
        self,
        model: str,
        collection: str,
        db_dir: str,
        n_results: int,
        openai_key: Optional[str] = None,
    ):
        if model == "openai":
            self.embed_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_key, model_name="text-embedding-ada-002"
            )
        else:
            self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model
            )

        self.n_results = int(n_results)
        self.client = chromadb.PersistentClient(
            path=db_dir,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self.collection = self.get_or_create_collection(collection)

    def get_or_create_collection(self, name: str):
        return self.client.get_or_create_collection(
            name=name,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_texts(self, texts: list[str], metadatas: list[dict]):
        offset = self.collection.count()
        ids = [str(offset + i) for i in range(len(texts))]
        self.collection.add(documents=texts, metadatas=metadatas, ids=ids)

    def add_embeddings(
        self, texts: list[str], embeddings: list[list], metadatas: list[dict]
    ):
        offset = self.collection.count()
        ids = [str(offset + i) for i in range(len(texts))]
        self.collection.add(
            documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids
        )

    def query(self, text: str):
        return self.collection.query(query_texts=[text], n_results=self.n_results)


class VectorMatch(BaseModel):
    text: str = ""
    metadata: Optional[dict] = {}
    distance: float = 0.0


class VectorDBScanner(AbstractDefence):
    def __init__(self, db_client: VectorDB, threshold: float):
        self.db_client: VectorDB = db_client
        self.threshold: float = float(threshold)

    @override
    def analyse(self, query: str) -> AnalysisResult:
        matches = self.db_client.query(query)
        vulnerabilities: list[str] = []
        existing_texts: set[str] = set()

        documents = matches.get("documents", [[]])[0]
        metadatas = matches.get("metadatas", [[]])[0]
        distances = matches.get("distances", [[]])[0]

        for text, metadata, distance in zip(documents, metadatas, distances):
            if distance < self.threshold and text not in existing_texts:
                m = VectorMatch(text=text, metadata=metadata, distance=distance)

                description = (
                    f"Query is semantically similar to a known threat. "
                    f"Matched Text: '{m.text}', "
                    f"Distance: {m.distance:.4f}, "
                    f"Threshold: {self.threshold}"
                )
                vulnerabilities.append(description)
                existing_texts.add(m.text)

        if not vulnerabilities:
            return AnalysisResult(
                type="No similar threats found in vector database.", is_safe=True
            )
        else:
            problem_string: str = "\n---\n".join(vulnerabilities)
            return AnalysisResult(type=problem_string, is_safe=False)
