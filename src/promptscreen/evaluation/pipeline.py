import json
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


def evaluate(cfg: DictConfig, guards: dict[str, AbstractDefence]) -> None:
    query_agent = QueryAgent("gpt-oss:20b")
    evaluator = AttackEvaluator("gpt-oss:20b", 0.1)
    metrics_calc = MetricsCalculator()
    scanner = OutputScanner() if OutputScanner is not None else None

    print(f"Starting Pipeline Evaluation with {len(guards)} active defences...")

    with open("offence/metrics_test_set.json") as fh:
        data = json.load(fh)

    count = 0
    total_time: float = 0.0
    for entry in data:
        if entry["classification"] == "benign":
            continue

        prompt: str = entry["prompt"]
        deemed_unsafe = False
        start_time = time.time()
        for _name, guard in guards.items():
            result = guard.analyse(prompt)

            if not result.get_verdict():
                deemed_unsafe = True
                break

        end_time = time.time()

        if deemed_unsafe:
            failed_result = evaluator.create_failed_attempt(prompt)
            metrics_calc.add_result(failed_result)

        else:
            response = query_agent.query(prompt)
            response = (
                scanner.scan_output(response) if scanner is not None else response
            )
            attack_result = evaluator.evaluate(end_time - start_time, response, prompt)
            metrics_calc.add_result(attack_result)
        count += 1
        total_time += time.time() - start_time

    metrics_calc.evaluate()
    if count != 0:
        print(f"Average time taken: {total_time / count}")
