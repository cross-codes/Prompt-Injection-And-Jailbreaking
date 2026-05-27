import logging
import os
from typing import Optional

import yara
from typing_extensions import override

from ..data import RULES_DIR
from .abstract_defence import AbstractDefence
from .ds.analysis_result import AnalysisResult

logger = logging.getLogger(__name__)


class Scanner(AbstractDefence):
    def __init__(self, rules_dir: Optional[str] = None):
        if rules_dir is None:
            # Use bundled rules by default
            rules_dir = str(RULES_DIR)

        if not os.path.exists(rules_dir) or not os.path.isdir(rules_dir):
            raise FileNotFoundError(
                f"YARA rules directory '{rules_dir}' not found or invalid."
            )

        self.rules_dir: str = rules_dir
        self.compiled_rules: Optional[yara.Rules] = None
        self._load_rules()

    def _load_rules(self) -> None:
        yara_paths = {}
        for file in os.listdir(self.rules_dir):
            if file.lower().endswith((".yar", ".yara")):
                yara_paths[file] = os.path.join(self.rules_dir, file)

        if not yara_paths:
            raise FileNotFoundError(f"No YARA rule files found in '{self.rules_dir}'.")

        try:
            self.compiled_rules = yara.compile(filepaths=yara_paths)
        except Exception as e:
            raise RuntimeError(f"Failed to compile YARA rules: {e}") from e

    @override
    def analyse(self, query: str) -> AnalysisResult:
        if not query.strip():
            return AnalysisResult("YARA scanner", True)

        if self.compiled_rules is None:
            return AnalysisResult("YARA scanner not initialized", False)

        try:
            matches = self.compiled_rules.match(data=query)
        except Exception:
            logger.exception("YARA matching failed for query; treating as inconclusive")
            return AnalysisResult("YARA scanner error: match failed", False)

        if matches:
            matched_rules = ", ".join([match.rule for match in matches])
            return AnalysisResult(
                f"YARA scanner found matched rules: {matched_rules}", False
            )

        return AnalysisResult("YARA scanner found no matches", True)
