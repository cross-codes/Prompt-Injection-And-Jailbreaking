"""Run ground truth evaluation (stats mode).

This evaluates guards against a labeled dataset and calculates
precision, recall, specificity, and accuracy.
"""

import hydra
from guard_setup import initialize_guards
from omegaconf import DictConfig

from promptscreen.evaluation import run_suite


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=" * 60)
    print("PROMPTSCREEN - Ground Truth Evaluation (Stats Mode)")
    print("=" * 60)

    guards = initialize_guards(cfg)

    if not guards:
        print("Error: No guards initialized. Check your config.")
        return

    print(f"\nEvaluating {len(guards)} guards...")
    run_suite(cfg, guards)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
