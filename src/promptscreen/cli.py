"""Command-line interface for PromptScreen."""

import json
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import click

from . import __version__
from .defence import (
    AbstractDefence,
    ClassifierCluster,
    EncodingDetectorGuard,
    HeuristicVectorAnalyzer,
    InjectionScanner,
    JailbreakInferenceAPI,
    PromptLengthGuard,
    RateLimitGuard,
    Scanner,
    ShieldGemma2BClassifier,
    UnicodeGuard,
    VectorDB,
    VectorDBScanner,
)

# Type for guard factory
GuardFactory = Callable[[], AbstractDefence]

# Guard registry with factory functions
AVAILABLE_GUARDS: dict[str, dict[str, Any]] = {
    "heuristic": {
        "factory": lambda: HeuristicVectorAnalyzer(threshold=2, pm_shot_lim=3),
        "name": "HeuristicVectorAnalyzer",
        "speed": "Very Fast (< 1ms)",
        "description": "Pattern-based detection using keyword matching",
        "requires": "core",
    },
    "scanner": {
        "factory": lambda: Scanner(),
        "name": "Scanner (YARA)",
        "speed": "Very Fast (< 5ms)",
        "description": "Pattern matching using bundled YARA rules",
        "requires": "core",
    },
    "injection": {
        "factory": lambda: InjectionScanner(),
        "name": "InjectionScanner",
        "speed": "Very Fast (< 1ms)",
        "description": "Regex-based detection of injection attempts",
        "requires": "core",
    },
    "length": {
        "factory": lambda: PromptLengthGuard(),
        "name": "PromptLengthGuard",
        "speed": "Very Fast (< 1ms)",
        "description": "Hard/soft character-count limits; blocks unusually long prompts",
        "requires": "core",
    },
    "unicode": {
        "factory": lambda: UnicodeGuard(),
        "name": "UnicodeGuard",
        "speed": "Very Fast (< 1ms)",
        "description": "Detects RTL overrides, zero-width chars, and homograph attacks",
        "requires": "core",
    },
    "encoding": {
        "factory": lambda: EncodingDetectorGuard(),
        "name": "EncodingDetectorGuard",
        "speed": "Very Fast (< 1ms)",
        "description": "Detects Base64, hex, ROT13, and URL-encoded payloads",
        "requires": "core",
    },
    "ratelimit": {
        "factory": lambda: RateLimitGuard(),
        "name": "RateLimitGuard",
        "speed": "Very Fast (< 1ms)",
        "description": "Sliding-window rate limiter; detects automated probing",
        "requires": "core",
    },
    "svm": {
        "factory": lambda: JailbreakInferenceAPI("model_artifacts"),
        "name": "JailbreakInferenceAPI (SVM)",
        "speed": "Medium (5-15ms)",
        "description": "ML classifier trained on jailbreak datasets",
        "requires": "core + trained model",
    },
    "vectordb": {
        "factory": lambda: _create_vectordb_guard(),
        "name": "VectorDBScanner",
        "speed": "Slow (50-200ms)",
        "description": "Similarity search against known threats",
        "requires": "[vectordb]",
    },
    "cluster": {
        "factory": lambda: ClassifierCluster(),
        "name": "ClassifierCluster",
        "speed": "Very Slow (1-3s)",
        "description": "Dual ML models (toxicity + jailbreak)",
        "requires": "[ml]",
    },
    "shieldgemma": {
        "factory": lambda: ShieldGemma2BClassifier(token=_get_hf_token()),
        "name": "ShieldGemma2BClassifier",
        "speed": "Very Slow (1-3s)",
        "description": "Google's ShieldGemma 2B safety model",
        "requires": "[ml] + HF token",
    },
}


def _create_vectordb_guard() -> "VectorDBScanner":  # type: ignore[name-defined]
    """Create VectorDB guard with default configuration."""
    if VectorDB is None:
        raise ImportError(
            "VectorDB requires chromadb. Install with: pip install promptscreen[vectordb]"
        )
    # For CLI, create empty DB - user should populate separately
    db = VectorDB(
        model="all-MiniLM-L6-v2", collection="threats", db_dir="chroma_db", n_results=5
    )
    return VectorDBScanner(db, threshold=0.3)


def _get_hf_token() -> str:
    """Get HuggingFace token from environment."""
    import os

    token = os.getenv("HUGGING_FACE_TOKEN")
    if not token:
        raise ValueError("ShieldGemma requires HUGGING_FACE_TOKEN environment variable")
    return token


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """PromptScreen - LLM prompt security toolkit.

    Detect prompt injection and jailbreak attempts in LLM inputs.

    \b
    Examples:
        promptscreen scan "Ignore all instructions"
        promptscreen scan --file prompts.txt --json
        promptscreen list-guards
        promptscreen compare "test prompt"
    """
    pass


@cli.command()
@click.argument("prompts", nargs=-1)
@click.option(
    "--file",
    "-f",
    type=click.Path(exists=True, path_type=Path),
    help="Read prompts from file (one per line)",
)
@click.option(
    "--guards",
    "-g",
    default="heuristic,scanner",
    show_default=True,
    help="Comma-separated list of guards to use",
)
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option(
    "--strict",
    is_flag=True,
    help="Exit with code 1 if any prompt is blocked",
)
def scan(
    prompts: tuple[str, ...],
    file: Optional[Path],
    guards: str,
    output_json: bool,
    verbose: bool,
    strict: bool,
) -> None:
    """Scan prompts for security threats.

    \b
    Examples:
        # Scan a single prompt
        promptscreen scan "Ignore all instructions"

        # Scan multiple prompts
        promptscreen scan "prompt1" "prompt2" "prompt3"

        # Scan from file
        promptscreen scan --file prompts.txt

        # Use specific guards
        promptscreen scan "test" --guards heuristic,svm

        # JSON output (for scripting)
        promptscreen scan "test" --json

        # Fail on unsafe prompts (for CI/CD)
        promptscreen scan "test" --strict
    """
    # Collect prompts
    prompt_list = list(prompts)
    if file:
        with open(file, encoding="utf-8") as f:
            prompt_list.extend(line.strip() for line in f if line.strip())

    if not prompt_list:
        click.echo("Error: No prompts provided", err=True)
        click.echo('Usage: promptscreen scan "prompt" or --file prompts.txt', err=True)
        sys.exit(1)

    # Initialize guards
    guard_names = [g.strip() for g in guards.split(",")]
    initialized_guards = {}

    for name in guard_names:
        if name not in AVAILABLE_GUARDS:
            click.echo(f"Error: Unknown guard '{name}'", err=True)
            click.echo(f"Available: {', '.join(AVAILABLE_GUARDS.keys())}")
            sys.exit(1)

        try:
            initialized_guards[name] = AVAILABLE_GUARDS[name]["factory"]()  # type: ignore
            if verbose:
                click.echo(f"✓ Initialized guard: {name}", err=True)
        except ImportError as e:
            click.echo(
                f"Error: Guard '{name}' requires additional dependencies", err=True
            )
            click.echo(f"  {e}", err=True)
            click.echo(
                f"  Install with: pip install promptscreen{AVAILABLE_GUARDS[name]['requires'].replace('core + trained model', '')}",
                err=True,
            )
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error initializing '{name}': {e}", err=True)
            sys.exit(1)

    # Scan prompts
    results: list[dict[str, Any]] = []
    any_blocked = False

    for prompt in prompt_list:
        prompt_result: dict[str, Any] = {"prompt": prompt, "guards": {}}

        for name, guard in initialized_guards.items():
            try:
                analysis = guard.analyse(prompt)
                is_safe = analysis.get_verdict()
                prompt_result["guards"][name] = {
                    "safe": is_safe,
                    "reason": analysis.get_type(),
                }  # type: ignore
                if not is_safe:
                    any_blocked = True
            except Exception as e:
                prompt_result["guards"][name] = {
                    "safe": None,
                    "error": str(e),
                }  # type: ignore

        results.append(prompt_result)

    # Output results
    if output_json:
        click.echo(json.dumps(results, indent=2))
    else:
        _print_results(results, verbose)

    # Exit with error if strict mode and any blocked
    if strict and any_blocked:
        sys.exit(1)


def _print_results(results: list[dict], verbose: bool) -> None:
    """Pretty-print scan results."""
    for i, result in enumerate(results, 1):
        prompt = result["prompt"]
        if len(prompt) > 60:
            prompt = prompt[:57] + "..."

        click.echo(f"\n[{i}] Prompt: {prompt}")

        all_safe = all(g.get("safe", False) for g in result["guards"].values())

        if all_safe:
            click.secho("  ✓ SAFE", fg="green", bold=True)
        else:
            click.secho("  ✗ BLOCKED", fg="red", bold=True)

        if verbose or not all_safe:
            for guard_name, guard_result in result["guards"].items():
                if guard_result.get("safe"):
                    click.echo(f"    {guard_name:12s}: ✓ safe")
                elif guard_result.get("error"):
                    click.secho(
                        f"    {guard_name:12s}: ERROR - {guard_result['error']}",
                        fg="yellow",
                    )
                else:
                    reason = guard_result.get("reason", "unknown")
                    click.secho(f"    {guard_name:12s}: ✗ {reason}", fg="red")


@cli.command("list-guards")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed information")
def list_guards(verbose: bool) -> None:
    """List all available guards.

    \b
    Example:
        promptscreen list-guards
        promptscreen list-guards --verbose
    """
    if verbose:
        click.echo("\nAvailable Guards:\n")
        for name, info in AVAILABLE_GUARDS.items():
            click.echo(f"  • {name}")
            click.echo(f"    Name:        {info['name']}")
            click.echo(f"    Speed:       {info['speed']}")
            click.echo(f"    Description: {info['description']}")
            click.echo(f"    Requires:    {info['requires']}")
            click.echo()
    else:
        click.echo("\nAvailable Guards:\n")
        for name in AVAILABLE_GUARDS.keys():
            click.echo(f"  • {name}")
        click.echo("\nUsage: promptscreen scan 'text' --guards heuristic,scanner")
        click.echo("Info:  promptscreen info <guard-name>\n")


@cli.command()
@click.argument("guard_name")
def info(guard_name: str) -> None:
    """Show detailed information about a guard.

    \b
    Example:
        promptscreen info heuristic
        promptscreen info svm
    """
    if guard_name not in AVAILABLE_GUARDS:
        click.echo(f"Error: Unknown guard '{guard_name}'", err=True)
        click.echo(f"Available: {', '.join(AVAILABLE_GUARDS.keys())}")
        sys.exit(1)

    info_dict = AVAILABLE_GUARDS[guard_name]

    click.echo(f"\n{info_dict['name']}")
    click.echo("=" * 60)
    click.echo(f"Speed:       {info_dict['speed']}")
    click.echo(f"Description: {info_dict['description']}")
    click.echo(f"Requires:    {info_dict['requires']}")

    # Additional details
    details = {
        "heuristic": [
            "Detects 'ignore' instructions",
            "Role-play attempts",
            "Urgency manipulation",
            "Hypothetical scenarios",
        ],
        "scanner": [
            "API tokens and secrets",
            "System instruction manipulation",
            "Command injection patterns",
            "IP addresses and URLs",
        ],
        "injection": [
            "DNS exfiltration (nslookup, dig)",
            "Markdown image exfiltration",
            "Backtick-wrapped commands",
        ],
        "svm": [
            "Known jailbreak patterns",
            "Sophisticated attack attempts",
            "ML-based classification",
        ],
        "vectordb": [
            "Similarity to known attacks",
            "Semantic understanding",
            "Requires threat database",
        ],
        "cluster": [
            "Toxicity detection",
            "Jailbreak classification",
            "Dual-model approach",
        ],
        "shieldgemma": [
            "Google's safety model",
            "Multi-category detection",
            "Production-grade accuracy",
        ],
    }

    if guard_name in details:
        click.echo("\nDetects:")
        for item in details[guard_name]:
            click.echo(f"  • {item}")

    click.echo()


@cli.command()
@click.argument("prompt")
@click.option(
    "--guards",
    default="heuristic,scanner,svm",
    show_default=True,
    help="Guards to compare",
)
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def compare(prompt: str, guards: str, output_json: bool) -> None:
    """Compare guard results side-by-side.

    \b
    Example:
        promptscreen compare "test prompt"
        promptscreen compare "test" --guards heuristic,svm,scanner
        promptscreen compare "test" --json
    """
    guard_names = [g.strip() for g in guards.split(",")]
    results = {}

    if not output_json:
        click.echo(f"\nComparing guards on: {prompt}\n")
        click.echo(f"{'Guard':<15} {'Result':<10} {'Reason'}")
        click.echo("-" * 70)

    for name in guard_names:
        if name not in AVAILABLE_GUARDS:
            if output_json:
                results[name] = {"error": "Unknown guard"}
            else:
                click.echo(f"{name:<15} ERROR      Unknown guard")
            continue

        try:
            guard = AVAILABLE_GUARDS[name]["factory"]()
            analysis = guard.analyse(prompt)

            is_safe = analysis.get_verdict()
            reason = analysis.get_type()

            if output_json:
                results[name] = {"safe": is_safe, "reason": reason}
            else:
                if is_safe:
                    result_str = click.style("✓ SAFE", fg="green")
                else:
                    result_str = click.style("✗ BLOCKED", fg="red")

                click.echo(f"{name:<15} {result_str:<19} {reason}")

        except Exception as e:
            if output_json:
                results[name] = {"error": str(e)}
            else:
                click.echo(f"{name:<15} ERROR      {str(e)}")

    if output_json:
        click.echo(json.dumps({"prompt": prompt, "results": results}, indent=2))
    else:
        click.echo()


@cli.command()
@click.option(
    "--guards",
    default="heuristic,scanner",
    show_default=True,
    help="Guards to use",
)
def interactive(guards: str) -> None:
    """Interactive prompt scanning mode.

    Enter prompts interactively and see results in real-time.
    Press Ctrl+C or Ctrl+D to exit.

    \b
    Example:
        promptscreen interactive
        promptscreen interactive --guards heuristic,svm,scanner
    """
    guard_names = [g.strip() for g in guards.split(",")]
    initialized_guards = {}

    # Initialize guards
    click.echo("Initializing guards...")
    for name in guard_names:
        if name not in AVAILABLE_GUARDS:
            click.echo(f"  ✗ {name}: Unknown guard", err=True)
            continue

        try:
            initialized_guards[name] = AVAILABLE_GUARDS[name]["factory"]()
            click.echo(f"  ✓ {name}")
        except Exception as e:
            click.echo(f"  ✗ {name}: {e}")

    if not initialized_guards:
        click.echo("Error: No guards initialized", err=True)
        sys.exit(1)

    click.echo("\nInteractive mode - Type prompts to scan (Ctrl+C to exit)\n")

    try:
        while True:
            try:
                prompt = click.prompt("Prompt", type=str)
            except (EOFError, KeyboardInterrupt):
                break

            if not prompt.strip():
                continue

            all_safe = True
            for name, guard in initialized_guards.items():
                try:
                    analysis = guard.analyse(prompt)
                    if analysis.get_verdict():
                        click.echo(f"  {name:12s}: ✓ safe")
                    else:
                        click.secho(f"  {name:12s}: ✗ {analysis.get_type()}", fg="red")
                        all_safe = False
                except Exception as e:
                    click.secho(f"  {name:12s}: ERROR - {e}", fg="yellow")

            if all_safe:
                click.secho("\n✓ SAFE\n", fg="green", bold=True)
            else:
                click.secho("\n✗ BLOCKED\n", fg="red", bold=True)

    except KeyboardInterrupt:
        pass

    click.echo("\nExiting...")


if __name__ == "__main__":
    cli()
