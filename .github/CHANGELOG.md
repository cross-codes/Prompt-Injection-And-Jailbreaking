# Changelog

All notable changes to this project will be documented in this file.

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- **`PromptLengthGuard`** (`length`) — hard-blocks prompts above a configurable character limit; soft-warns with reduced confidence above a lower threshold. Defends against many-shot, padding, and context-overflow attacks.
- **`UnicodeGuard`** (`unicode`) — detects RTL override characters (U+202E etc.), zero-width steganography (U+200B/C/D, FEFF), and mixed-script homograph attacks (e.g. Cyrillic 'а' mixed with Latin text).
- **`EncodingDetectorGuard`** (`encoding`) — decodes and inspects Base64 blobs, hex dumps (spaced and compact), ROT13 strings, and URL percent-encoded sequences; blocks when decoded content contains injection keywords. `block_on_encoding_alone=True` mode blocks any encoded blob regardless of content.
- **`RateLimitGuard`** (`ratelimit`) — thread-safe sliding-window rate limiter; exposes `remaining()` and `reset()` helpers. Detects and blocks automated jailbreak probing.
- **`AnalysisResult.confidence`** field — optional `float` (0.0–1.0) surfacing guard certainty. `InjectionScanner` populates it with the average confidence of all found vulnerabilities; new guards all populate it.
- **`IMPROVEMENTS.md`** — improvement roadmap covering planned safety layers, new attack types, and SVM enhancements.
- 68 new unit tests for the four new guards (boundary, positive, negative, thread-safety).

### Changed

- **`AnalysisResult.type` renamed to `reason`** — `type` shadowed Python's built-in. `get_type()` kept as a backward-compatible alias; all internal call sites updated.
- **API chain mode** — `POST /evaluate` with `mode: "chain"` now always returns `{guard_name: DefenceResult}` for every evaluated guard (early-stops on first block). Previously returned an inconsistent `{"ChainResult": ...}` key on success vs `{guard_name: ...}` on failure.
- **Pre-commit config** — replaced `black` + `ruff --fix` with `ruff-format` (formatter) + `ruff` (linter). Eliminates the formatter conflict where Ruff's `I`/`Q` fixes rewrote Black's output. `ruff-format` is Black-compatible; output is identical.
- **`pydantic`** added to `[all]` extras in `pyproject.toml`.
- **`disallow_untyped_defs = true`** enabled in mypy config; all missing return-type annotations added across ~12 files.
- All `print()` debug/info calls replaced with `logging.getLogger(__name__)` across `pipeline.py`, `ds_metrics.py`, `train_classifier.py`, `scanner.py`, and `api/server.py`.
- **`TextPreProcessor`** extracted to `src/promptscreen/utils/text_preprocessor.py`; removed duplicate implementations in `linear_svm.py` and `train_classifier.py`.
- **pytest markers** auto-applied via `conftest.py` hook — `tests/unit/` → `@pytest.mark.unit`, `tests/integration/` → `@pytest.mark.integration`. Fixes `0 selected` when running `pytest -m unit`.
- **`RateLimitGuard`** registered in CLI (`ratelimit`), `defence/__init__.py`, and `__all__`.

### Fixed

- **`UnboundLocalError` in `pipeline.py`** — `end_time` was only assigned inside a guard loop; moved outside so it is always defined even when no guards run.
- **Broken JSON fence-stripping in `metrics_classes.py`** — `_safe_json_loads` incorrectly sliced the opening fence and then tried to `replace("```json", "")` on already-sliced content where that string cannot appear. Rewritten using `find("\n")` + `rfind("```")`.
- **VectorDB ID collision** — `add_texts` and `add_embeddings` used `range(len(texts))` for IDs, causing collisions on repeated inserts. Now uses `self.collection.count()` as offset.
- **Debug `print` in `linear_svm.py`** — removed `print(model_path, feature_union_path, "Testing paths")`.
- **Silent exception swallow in `scanner.py`** — YARA errors were caught and returned `is_safe=True`. Now logs via `logging.exception()` and returns `is_safe=False`.
- **Punctuation-blind keyword matching in `heuristic_channel.py`** — `"urgent!"` and `"ignore,"` failed to match because punctuation was not stripped before splitting. Fixed with `str.translate`.
- **Duplicate SEPARATORS in `ppa_defence.py`** — 12 duplicate entries removed (84 → 72 unique separator pairs).
- **Hardcoded `"gpt-oss:20b"` model in `pipeline.py`** — now reads `cfg.get("llm_model", "llama3")` from Hydra config; `llm_model` key added to `conf/config.yaml`.
- **Benign prompts skipped in pipeline** — benign prompts are now run through all guards and false-positive rate is tracked and printed.
- **Dead code in `train_classifier.py`** — removed `_train_model()` (never called) and `_load_and_preprocess_data()`. Fixed unknown-label handling to `dropna()` rather than only warning.
- **`api/server.py` error handling** — extracted `_run_guard()` helper with `try/except`; guard exceptions return structured `is_safe=False` response with logged stack trace instead of crashing with HTTP 500.
- **ASR denominator in `metrics_classes.py`** — confirmed `calculate_asr()` correctly divides by `total_attempted` (non-blocked attacks), not `len(self.attack_results)`.
- **`shieldgemma.py` formatting** — file was not Black/ruff-format compliant; reformatted.

---

## [0.3.0] - 2026-01-02

### Added

- **CLI** - `promptscreen scan "Ignore all instructions"` now works!
- Check `src/promptscreen/cli.py` for more details!
- Added tests for CLI

### Fixed

- Added tests/ to CI
- Re-ordered imports to fix ruff errors
- Bumped conflicting versions of ruff linter

## [0.3.0]: https://github.com/dronefreak/PromptScreen/releases/tag/v0.3.0

---

## [0.2.0] - 2025-12-26

### Added

- **PyPI package publication** - `pip install promptscreen` now works!
- Properly configured packaging for distribution

### Fixed

- typing-extensions dependency now installed on all Python versions (fixes Python 3.12 import error)
- VectorDB and ML guards now properly optional (lazy imports)
- chromadb import error when using core package only

### Changed

- First public release on PyPI (previously source-only)
- Improved optional dependency handling

## [0.2.0]: https://github.com/dronefreak/PromptScreen/releases/tag/v0.2.0

---

## [0.1.0] - 2025-12-25

### Added

- Initial public release of **PromptScreen**
- Prompt injection and jailbreak detection via:
  - Heuristic and regex-based scanners
  - YARA rule matching
  - Optional ML-based classifiers
  - Optional vector similarity detection
- Modular guard architecture with independently usable components
- FastAPI-based API server with multiple evaluation modes
- Evaluation framework for measuring attack success rate (ASR)

### Notes

- This is an **alpha release**; APIs may change
- Package is not yet published to PyPI
- ML-based guards have limited test coverage

### Added

- **CLI** - `promptscreen scan "Ignore all instructions"` now works!
- Check `src/promptscreen/cli.py` for more details!
- Added tests for CLI

### Fixed

- Added tests/ to CI
- Re-ordered imports to fix ruff errors
- Bumped conflicting versions of ruff linter

## [0.3.0]: https://github.com/dronefreak/PromptScreen/releases/tag/v0.3.0

---

## [0.2.0] - 2025-12-26

### Added

- **PyPI package publication** - `pip install promptscreen` now works!
- Properly configured packaging for distribution

### Fixed

- typing-extensions dependency now installed on all Python versions (fixes Python 3.12 import error)
- VectorDB and ML guards now properly optional (lazy imports)
- chromadb import error when using core package only

### Changed

- First public release on PyPI (previously source-only)
- Improved optional dependency handling

## [0.2.0]: https://github.com/dronefreak/PromptScreen/releases/tag/v0.2.0

---

## [0.1.0] - 2025-12-25

### Added

- Initial public release of **PromptScreen**
- Prompt injection and jailbreak detection via:
  - Heuristic and regex-based scanners
  - YARA rule matching
  - Optional ML-based classifiers
  - Optional vector similarity detection
- Modular guard architecture with independently usable components
- FastAPI-based API server with multiple evaluation modes
- Evaluation framework for measuring attack success rate (ASR)

### Notes

- This is an **alpha release**; APIs may change
- Package is not yet published to PyPI
- ML-based guards have limited test coverage
