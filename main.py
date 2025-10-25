import json, os
from typing import Any, TextIO
from defence.classifier_cluster import ClassifierCluster
from dotenv import load_dotenv
from metrics import AttackEvaluator, AttackResult, MetricsCalculator
from defence.heuristic_channel import HeuristicVectorAnalyzer
from defence.shieldgemma import ShieldGemma2BClassifier
from query_agent import QueryAgent

if __name__ == "__main__":
    _ = load_dotenv()
    huggingface_token: str | None = os.getenv("HUGGING_FACE_TOKEN")

    query_agent: QueryAgent = QueryAgent("tinyllama")
    cluster_guard: ClassifierCluster = ClassifierCluster()
    heuristic_guard: HeuristicVectorAnalyzer = HeuristicVectorAnalyzer(3, 3)
    llm_guard: ShieldGemma2BClassifier = ShieldGemma2BClassifier(huggingface_token)

    evaluator: AttackEvaluator = AttackEvaluator("tinyllama", 0.1)
    metrics_calc: MetricsCalculator = MetricsCalculator()

    def calculate_and_write_metrics(
        data: list[dict[str, Any]],
        guard,
        guard_label: str,
        output_file: TextIO,
    ):
        tp, fn, fp, tn = 0, 0, 0, 0
        print(f"Processing metrics for: {guard_label}...")
        for entry in data:
            classification: bool = entry["classification"] == "benign"
            result: bool = guard.is_safe(entry["prompt"])
            if classification and result:
                tp += 1
            elif classification and not result:
                fn += 1
            elif not classification and result:
                fp += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

        _ = output_file.write(f"--- {guard_label} Metrics ---\n")
        _ = output_file.write(f"Precision: {precision}\n")
        _ = output_file.write(f"Sensitivity (Recall): {sensitivity}\n")
        _ = output_file.write(f"Specificity: {specificity}\n")
        _ = output_file.write(f"Negative Predictive Value: {npv}\n")
        _ = output_file.write(f"Accuracy: {accuracy}\n\n")

    INPUT_FILE = "offence/combined_classified_prompts.json"
    OUTPUT_FILE = "all_metrics_results.txt"

    guards = {
        "Heuristic": heuristic_guard,
        "Classifier Cluster": cluster_guard,
        "ShieldGemma": llm_guard,
    }

    with open(INPUT_FILE, "r") as fh_in:
        data_to_process: list[dict[str, Any]] = json.load(fh_in)

    with open(OUTPUT_FILE, "w") as fh_out:
        for label, guard_instance in guards.items():
            calculate_and_write_metrics(data_to_process, guard_instance, label, fh_out)

