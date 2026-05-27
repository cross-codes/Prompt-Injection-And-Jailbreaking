import json
import logging
import time

from omegaconf import DictConfig

from ..defence.abstract_defence import AbstractDefence
from ..utils.query_agent import QueryAgent
from .metrics_classes import AttackEvaluator, MetricsCalculator

try:
    from ..output_scanners.scan import OutputScanner

    _has_output_scanner = True
except ImportError:
    OutputScanner = None  # type: ignore
    _has_output_scanner = False

logger = logging.getLogger(__name__)
_DEFAULT_MODEL = "llama3"


def evaluate(cfg: DictConfig, guards: dict[str, AbstractDefence]) -> None:
    model_name: str = cfg.get("llm_model", _DEFAULT_MODEL)  # type: ignore[arg-type]
    query_agent = QueryAgent(model_name)
    evaluator = AttackEvaluator(model_name, 0.1)
    metrics_calc = MetricsCalculator()
    scanner = OutputScanner() if OutputScanner is not None else None

    logger.info("Starting Pipeline Evaluation with %d active defences...", len(guards))
    logger.info("Using LLM model: %s", model_name)

    with open(cfg.get("test_file", "offence/metrics_test_set.json")) as fh:  # type: ignore[arg-type]
        data = json.load(fh)

    attack_count = 0
    benign_count = 0
    false_positive_count = 0
    total_time: float = 0.0

    for entry in data:
        classification: str = entry["classification"]
        prompt: str = entry["prompt"]
        is_benign = classification == "benign"

        start_time = time.time()
        deemed_unsafe = False
        for _name, guard in guards.items():
            result = guard.analyse(prompt)
            if not result.get_verdict():
                deemed_unsafe = True
                break
        end_time = time.time()
        elapsed = end_time - start_time
        total_time += elapsed

        if is_benign:
            benign_count += 1
            if deemed_unsafe:
                false_positive_count += 1
            # Benign prompts are not sent to the LLM; they are only used to
            # track false-positive rate across the guard chain.
            continue

        # Malicious prompt path
        attack_count += 1
        if deemed_unsafe:
            failed_result = evaluator.create_failed_attempt(prompt)
            metrics_calc.add_result(failed_result)
        else:
            response = query_agent.query(prompt)
            response = (
                scanner.scan_output(response) if scanner is not None else response
            )
            attack_result = evaluator.evaluate(elapsed, response, prompt)
            metrics_calc.add_result(attack_result)

    metrics_calc.evaluate()

    total_count = attack_count + benign_count
    if total_count > 0:
        logger.info("Average time per prompt: %.4fs", total_time / total_count)

    if benign_count > 0:
        fpr = false_positive_count / benign_count
        logger.info(
            "FALSE POSITIVE RATE: %.2f%% (%d/%d benign prompts incorrectly blocked)",
            fpr * 100,
            false_positive_count,
            benign_count,
        )
