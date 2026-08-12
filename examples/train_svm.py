"""Train SVM classifier for jailbreak detection."""

import logging

from promptscreen.defence.train import JailbreakClassifier


def main():
    # JailbreakClassifier.train() reports its classification_report via the
    # logging module; without a configured handler, that output is silently
    # dropped and training looks like it produced no report at all.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("Training SVM classifier...")

    # Paths
    train_file = "offence/metrics_train_set.json"  # Adjust if needed
    model_dir = "model_artifacts"

    # Train
    trainer = JailbreakClassifier(json_file_path=train_file, model_output_dir=model_dir)

    print(f"\nTraining on: {train_file}")
    print(f"Output directory: {model_dir}")
    print("-" * 60)

    trainer.train()

    print("\n" + "=" * 60)
    print("✓ Training complete!")
    print(f"✓ Models saved to: {model_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
