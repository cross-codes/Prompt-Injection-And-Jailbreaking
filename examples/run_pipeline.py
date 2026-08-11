"""Run LLM judge evaluation (pipeline mode).

This evaluates end-to-end attack success by running prompts through guards,
sending to LLM, and judging if attack succeeded.
"""

import hydra
from guard_setup import initialize_guards
from omegaconf import DictConfig

from promptscreen.evaluation import evaluate


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=" * 60)
    print("PROMPTSCREEN - Pipeline Evaluation (LLM Judge Mode)")
    print("=" * 60)

    guards = initialize_guards(cfg)

    if not guards:
        print("Error: No guards initialized. Check your config.")
        return

    print(f"\nRunning pipeline with {len(guards)} guards...")
    evaluate(cfg, guards)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
