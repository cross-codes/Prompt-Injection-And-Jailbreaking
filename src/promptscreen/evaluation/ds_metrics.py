import json
import logging
from typing import Any, TextIO

from omegaconf import DictConfig

from ..defence.abstract_defence import AbstractDefence

logger = logging.getLogger(__name__)


def calculate_and_write_metrics(
    data: list[dict[str, Any]],
    guard: AbstractDefence,
    guard_label: str,
    output_file: TextIO,
) -> None:
    # Positive class = "malicious prompt, correctly blocked" (the standard
    # framing for an attack-detection system). tp/fn track attack prompts,
    # fp/tn track benign prompts.
    tp, fn, fp, tn = 0, 0, 0, 0
    logger.info("Processing metrics for: %s", guard_label)
    for entry in data:
        is_malicious: bool = entry["classification"] != "benign"
        is_blocked: bool = not guard.analyse(entry["prompt"]).get_verdict()

        if is_malicious and is_blocked:
            tp += 1  # attack correctly blocked
        elif is_malicious and not is_blocked:
            fn += 1  # attack missed
        elif not is_malicious and is_blocked:
            fp += 1  # benign prompt incorrectly blocked
        else:
            tn += 1  # benign prompt correctly allowed

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    _ = output_file.write(f"--- {guard_label} Metrics ---\n")
    _ = output_file.write(f"Precision: {precision:.4f}\n")
    _ = output_file.write(f"Sensitivity (Recall): {sensitivity:.4f}\n")
    _ = output_file.write(f"Specificity: {specificity:.4f}\n")
    _ = output_file.write(f"Negative Predictive Value: {npv:.4f}\n")
    _ = output_file.write(f"Accuracy: {accuracy:.4f}\n\n")


def run_suite(cfg: DictConfig, guards: dict) -> None:
    if "shieldgemma" in cfg.active_defences and not cfg.huggingface_token:
        raise ValueError(
            "ShieldGemma is active, but HUGGING_FACE_TOKEN is missing. "
            "Please set it in your environment, config file, or pass as a param"
        )

    with open(cfg.input_file) as fh_in:
        data_to_process: list[dict] = json.load(fh_in)

    open(cfg.output_file, "w").close()  # noqa: SIM115

    for label, guard_instance in guards.items():
        with open(cfg.output_file, "a") as fh_out:
            calculate_and_write_metrics(data_to_process, guard_instance, label, fh_out)

    logger.info("Results stored in '%s'", cfg.output_file)
