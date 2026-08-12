"""Near-duplicate-aware train/test split for the SVM training corpus.

The dataset is heavily templated (many LLM-generated variants of the same
scenario), so a naive random split has near-zero *exact*-string overlap
between train and test but heavy *near*-duplicate leakage -- template
siblings landing on both sides -- which inflates reported test accuracy.

This module groups near-duplicate prompts (via rare-shingle Jaccard
overlap) with union-find, then splits at the *group* level so an entire
template family lands entirely on one side of the split.
"""

import json
import random
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from ...utils.unicode_normalize import normalize_for_model

# `type` values in the source dataset that exist specifically to test
# obfuscation robustness (paired plain/obfuscated variants of the same
# jailbreak content, or adversarial-LLM-generated attacks). These are held
# out of the main train/test split entirely -- see build_near_duplicate_split.
ROBUSTNESS_TYPES = frozenset({"emoji", "regular", "advllm"})

_DEFAULT_SHINGLE_SIZE = 8
_DEFAULT_MAX_DF = 20
_DEFAULT_TEST_FRAC = 0.07
_DEFAULT_SEED = 42
_FALLBACK_SHINGLE_SIZE = 3
_FALLBACK_JACCARD_THRESHOLD = 0.6


class _UnionFind:
    def __init__(self, n: int) -> None:
        self._parent = list(range(n))

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path halving
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb

    def groups(self) -> list[int]:
        """Return a group id per original index (0..n-1)."""
        return [self.find(i) for i in range(len(self._parent))]


def _shingle_texts(texts: list[str], n: int) -> list[set[tuple[str, ...]]]:
    shingles: list[set[tuple[str, ...]]] = []
    for text in texts:
        words = text.split()
        if len(words) < n:
            shingles.append(set())
        else:
            shingles.append(
                {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}
            )
    return shingles


def _rare_shingle_edges(
    texts: list[str], shingle_size: int, max_df: int, chunk_size: int = 2000
) -> set[tuple[int, int]]:
    """Index pairs (i, j), i < j, sharing >=1 shingle with doc-frequency <= max_df."""
    vectorizer = CountVectorizer(
        analyzer="word",
        ngram_range=(shingle_size, shingle_size),
        binary=True,
        token_pattern=r"\S+",  # noqa: S106 # nosec B106 -- a regex pattern, not a credential
    )
    matrix = vectorizer.fit_transform(texts)
    if matrix.shape[1] == 0:
        return set()

    doc_freq = np.asarray(matrix.sum(axis=0)).ravel()
    rare_mask = doc_freq <= max_df
    if not rare_mask.any():
        return set()
    rare_matrix = matrix[:, rare_mask].tocsr()

    edges: set[tuple[int, int]] = set()
    n = rare_matrix.shape[0]
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        block = (rare_matrix[start:end] @ rare_matrix.T).tocoo()
        for r, c, v in zip(block.row, block.col, block.data):
            if v >= 1:
                i, j = start + r, c
                if i < j:
                    edges.add((i, j))
    return edges


def _fallback_short_prompt_edges(
    texts: list[str], candidate_indices: list[int]
) -> set[tuple[int, int]]:
    """Near-duplicate edges among prompts too short for the main shingle size.

    Uses an inverted index over 3-word shingles to find candidate pairs
    (avoiding an O(k^2) scan), then confirms with Jaccard similarity.
    Prompts too short even for a 3-gram are grouped by exact normalized text.
    """
    shingles = _shingle_texts(
        [texts[i] for i in candidate_indices], _FALLBACK_SHINGLE_SIZE
    )

    inverted: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for local_i, sh in enumerate(shingles):
        for gram in sh:
            inverted[gram].append(local_i)

    edges: set[tuple[int, int]] = set()
    seen_pairs: set[tuple[int, int]] = set()
    for members in inverted.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                pair = (min(members[a], members[b]), max(members[a], members[b]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                si, sj = shingles[pair[0]], shingles[pair[1]]
                union = len(si | sj)
                if union and len(si & sj) / union >= _FALLBACK_JACCARD_THRESHOLD:
                    gi, gj = candidate_indices[pair[0]], candidate_indices[pair[1]]
                    edges.add((gi, gj) if gi < gj else (gj, gi))

    # Exact normalized-text match for prompts too short for even a 3-gram.
    norm_text_groups: dict[str, list[int]] = defaultdict(list)
    for i in candidate_indices:
        norm_text_groups[texts[i].strip().lower()].append(i)
    for members in norm_text_groups.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                gi, gj = members[a], members[b]
                edges.add((gi, gj) if gi < gj else (gj, gi))

    return edges


def build_groups(
    prompts: list[str],
    shingle_size: int = _DEFAULT_SHINGLE_SIZE,
    max_df: int = _DEFAULT_MAX_DF,
) -> list[int]:
    """Group near-duplicate prompts. Returns a group id per prompt index."""
    uf = _UnionFind(len(prompts))

    # Raw-text shingling (catches templated/paraphrased duplicates).
    for i, j in _rare_shingle_edges(prompts, shingle_size, max_df):
        uf.union(i, j)

    # Normalized-text shingling (catches near-dups that differ only by the
    # kind of Unicode obfuscation normalize_for_model reverses).
    normalized = [normalize_for_model(p) for p in prompts]
    for i, j in _rare_shingle_edges(normalized, shingle_size, max_df):
        uf.union(i, j)

    # Short-prompt fallback: anything with fewer than `shingle_size` words
    # never appears in the shingle matrix above and would always be a
    # trivial singleton group otherwise.
    short_indices = [i for i, p in enumerate(prompts) if len(p.split()) < shingle_size]
    if short_indices:
        for i, j in _fallback_short_prompt_edges(prompts, short_indices):
            uf.union(i, j)

    return uf.groups()


def _group_stratified_split(
    classifications: list[str],
    group_ids: list[int],
    test_frac: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Greedily fill each class's test quota without splitting a group."""
    group_members: dict[int, list[int]] = defaultdict(list)
    for idx, g in enumerate(group_ids):
        group_members[g].append(idx)

    group_label: dict[int, str] = {}
    for g, members in group_members.items():
        counts = Counter(classifications[m] for m in members)
        group_label[g] = counts.most_common(1)[0][0]

    class_counts = Counter(classifications)
    test_quota = {c: round(test_frac * n) for c, n in class_counts.items()}

    rng = random.Random(
        seed
    )  # nosec B311 -- reproducible split shuffle, not security-sensitive
    ordered_groups = list(group_members.keys())
    rng.shuffle(ordered_groups)

    test_indices: list[int] = []
    test_class_counts: Counter = Counter()
    for g in ordered_groups:
        members = group_members[g]
        label = group_label[g]
        if test_class_counts[label] + len(members) <= test_quota[label]:
            test_indices.extend(members)
            test_class_counts[label] += len(members)

    test_set = set(test_indices)
    train_indices = [i for i in range(len(classifications)) if i not in test_set]
    return train_indices, test_indices


def compute_leakage(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    shingle_size: int = _DEFAULT_SHINGLE_SIZE,
) -> float:
    """Fraction of test prompts sharing >=1 shingle with some training prompt."""
    train_shingles: set[tuple[str, ...]] = set()
    for entry in train:
        train_shingles.update(_shingle_texts([entry["prompt"]], shingle_size)[0])

    if not test:
        return 0.0

    leaked = 0
    for entry in test:
        test_shingles = _shingle_texts([entry["prompt"]], shingle_size)[0]
        if test_shingles and (test_shingles & train_shingles):
            leaked += 1
    return leaked / len(test)


def build_near_duplicate_split(
    input_path: str,
    train_path: str,
    test_path: str,
    robustness_path: str,
    test_frac: float = _DEFAULT_TEST_FRAC,
    shingle_size: int = _DEFAULT_SHINGLE_SIZE,
    max_df: int = _DEFAULT_MAX_DF,
    seed: int = _DEFAULT_SEED,
    manifest_path: str | None = None,
) -> dict[str, Any]:
    """Rebuild train/test/robustness files from `input_path` (e.g. dedup.json).

    Returns a manifest dict describing the resulting split (also written to
    `manifest_path` if given).
    """
    with open(input_path, encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)

    robustness = [e for e in data if e.get("type") in ROBUSTNESS_TYPES]
    main_pool = [e for e in data if e.get("type") not in ROBUSTNESS_TYPES]

    prompts = [e["prompt"] for e in main_pool]
    classifications = [e["classification"] for e in main_pool]

    group_ids = build_groups(prompts, shingle_size=shingle_size, max_df=max_df)
    train_idx, test_idx = _group_stratified_split(
        classifications, group_ids, test_frac=test_frac, seed=seed
    )

    train_rows = [main_pool[i] for i in train_idx]
    test_rows = [main_pool[i] for i in test_idx]

    for path, rows in ((train_path, train_rows), (test_path, test_rows)):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

    with open(robustness_path, "w", encoding="utf-8") as f:
        json.dump(robustness, f, indent=2, ensure_ascii=False)

    leakage = compute_leakage(train_rows, test_rows, shingle_size=shingle_size)

    group_sizes = Counter(group_ids)
    manifest = {
        "seed": seed,
        "shingle_size": shingle_size,
        "max_df": max_df,
        "test_frac": test_frac,
        "input_path": input_path,
        "input_rows": len(data),
        "robustness_rows_excluded": len(robustness),
        "main_pool_rows": len(main_pool),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "train_class_counts": dict(Counter(classifications[i] for i in train_idx)),
        "test_class_counts": dict(Counter(classifications[i] for i in test_idx)),
        "num_groups": len(group_sizes),
        "largest_group": max(group_sizes.values()) if group_sizes else 0,
        "rows_in_multi_member_groups": sum(
            size for size in group_sizes.values() if size > 1
        ),
        "leakage_any_shingle": leakage,
    }

    if manifest_path:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    result = build_near_duplicate_split(
        input_path="offence/dedup.json",
        train_path="offence/metrics_train_set.json",
        test_path="offence/metrics_test_set.json",
        robustness_path="offence/robustness_eval.json",
        manifest_path="offence/split_manifest.json",
    )
    print(json.dumps(result, indent=2))
