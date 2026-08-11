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

# Type for guard factory. Receives a dict of CLI-supplied overrides (parsed
# from --option guard.param=value); guards that don't accept overrides just
# ignore it.
GuardFactory = Callable[[dict[str, Any]], AbstractDefence]

# Guard registry with factory functions. "options" documents the CLI-tunable
# constructor parameters and their defaults for `info`/`list-guards
# --verbose`; guards with no tunable parameters have an empty dict.
AVAILABLE_GUARDS: dict[str, dict[str, Any]] = {
    "heuristic": {
        "factory": lambda opts: HeuristicVectorAnalyzer(
            threshold=opts.get("threshold", 2), pm_shot_lim=opts.get("pm_shot_lim", 3)
        ),
        "name": "HeuristicVectorAnalyzer",
        "speed": "Very Fast (< 1ms)",
        "description": "Pattern-based detection using keyword matching",
        "requires": "core",
        "options": {"threshold": 2, "pm_shot_lim": 3},
    },
    "scanner": {
        "factory": lambda opts: Scanner(rules_dir=opts.get("rules_dir")),
        "name": "Scanner (YARA)",
        "speed": "Very Fast (< 5ms)",
        "description": "Pattern matching using bundled YARA rules",
        "requires": "core",
        "options": {"rules_dir": None},
    },
    "injection": {
        "factory": lambda opts: InjectionScanner(),
        "name": "InjectionScanner",
        "speed": "Very Fast (< 1ms)",
        "description": "Regex-based detection of injection attempts",
        "requires": "core",
        "options": {},
    },
    "length": {
        "factory": lambda opts: PromptLengthGuard(
            max_chars=opts.get("max_chars", 10_000),
            warn_chars=opts.get("warn_chars", 4_000),
        ),
        "name": "PromptLengthGuard",
        "speed": "Very Fast (< 1ms)",
        "description": "Hard/soft character-count limits; blocks unusually long prompts",
        "requires": "core",
        "options": {"max_chars": 10_000, "warn_chars": 4_000},
    },
    "unicode": {
        "factory": lambda opts: UnicodeGuard(min_word_len=opts.get("min_word_len", 4)),
        "name": "UnicodeGuard",
        "speed": "Very Fast (< 1ms)",
        "description": "Detects RTL overrides, zero-width chars, and homograph attacks",
        "requires": "core",
        "options": {"min_word_len": 4},
    },
    "encoding": {
        "factory": lambda opts: EncodingDetectorGuard(
            block_on_encoding_alone=opts.get("block_on_encoding_alone", False)
        ),
        "name": "EncodingDetectorGuard",
        "speed": "Very Fast (< 1ms)",
        "description": "Detects Base64, hex, ROT13, and URL-encoded payloads",
        "requires": "core",
        "options": {"block_on_encoding_alone": False},
    },
    "ratelimit": {
        "factory": lambda opts: RateLimitGuard(
            max_requests=opts.get("max_requests", 60),
            window_seconds=opts.get("window_seconds", 60.0),
        ),
        "name": "RateLimitGuard",
        "speed": "Very Fast (< 1ms)",
        "description": "Sliding-window rate limiter; detects automated probing",
        "requires": "core",
        "options": {"max_requests": 60, "window_seconds": 60.0},
    },
    "svm": {
        "factory": lambda opts: JailbreakInferenceAPI(
            opts.get("model_dir", "model_artifacts")
        ),
        "name": "JailbreakInferenceAPI (SVM)",
        "speed": "Medium (5-15ms)",
        "description": "ML classifier trained on jailbreak datasets",
        "requires": "core + trained model",
        "options": {"model_dir": "model_artifacts"},
    },
    "vectordb": {
        "factory": lambda opts: _create_vectordb_guard(opts),
        "name": "VectorDBScanner",
        "speed": "Slow (50-200ms)",
        "description": "Similarity search against known threats",
        "requires": "[vectordb]",
        "options": {
            "model": "all-MiniLM-L6-v2",
            "collection": "threats",
            "db_dir": "chroma_db",
            "n_results": 5,
            "threshold": 0.3,
        },
    },
    "cluster": {
        "factory": lambda opts: ClassifierCluster(),
        "name": "ClassifierCluster",
        "speed": "Very Slow (1-3s)",
        "description": "Dual ML models (toxicity + jailbreak)",
        "requires": "[ml]",
        "options": {},
    },
    "shieldgemma": {
        "factory": lambda opts: ShieldGemma2BClassifier(token=_get_hf_token()),
        "name": "ShieldGemma2BClassifier",
        "speed": "Very Slow (1-3s)",
        "description": "Google's ShieldGemma 2B safety model",
        "requires": "[ml] + HF token",
        # Deliberately not overridable via --option: the token is a secret
        # and shouldn't be passable on the command line / shell history.
        "options": {},
    },
}


def _create_vectordb_guard(opts: dict[str, Any]) -> "VectorDBScanner":  # type: ignore[name-defined]
    """Create VectorDB guard, optionally overridden by CLI --option values."""
    if VectorDB is None:
        raise ImportError(
            "VectorDB requires chromadb. Install with: pip install promptscreen[vectordb]"
        )
    # For CLI, create empty DB - user should populate separately
    db = VectorDB(
        model=opts.get("model", "all-MiniLM-L6-v2"),
        collection=opts.get("collection", "threats"),
        db_dir=opts.get("db_dir", "chroma_db"),
        n_results=opts.get("n_results", 5),
    )
    return VectorDBScanner(db, threshold=opts.get("threshold", 0.3))


def _get_hf_token() -> str:
    """Get HuggingFace token from environment."""
    import os

    token = os.getenv("HUGGING_FACE_TOKEN")
    if not token:
        raise ValueError("ShieldGemma requires HUGGING_FACE_TOKEN environment variable")
    return token


def _coerce_option_value(value: str) -> Any:
    """Best-effort string -> Python type coercion for --option values."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _parse_guard_options(raw_options: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Parse repeated ``--option guard.param=value`` flags.

    Returns ``{guard_name: {param: coerced_value}}``. Raises
    ``click.BadOptionUsage`` on malformed input.
    """
    overrides: dict[str, dict[str, Any]] = {}
    for raw in raw_options:
        if "=" not in raw:
            raise click.BadOptionUsage(
                "option",
                f"Invalid --option '{raw}'. Expected format: guard.param=value",
            )
        key, value = raw.split("=", 1)
        if "." not in key:
            raise click.BadOptionUsage(
                "option",
                f"Invalid --option '{raw}'. Expected format: guard.param=value",
            )
        guard_name, param = key.split(".", 1)
        overrides.setdefault(guard_name, {})[param] = _coerce_option_value(value)
    return overrides


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
@click.option(
    "--option",
    "-o",
    "raw_options",
    multiple=True,
    metavar="GUARD.PARAM=VALUE",
    help="Override a guard's constructor parameter, e.g. -o heuristic.threshold=3. "
    "Repeatable. See `promptscreen info <guard>` for tunable parameters.",
)
def scan(
    prompts: tuple[str, ...],
    file: Optional[Path],
    guards: str,
    output_json: bool,
    verbose: bool,
    strict: bool,
    raw_options: tuple[str, ...],
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

        # Tune a guard's parameters
        promptscreen scan "test" --guards length -o length.max_chars=500

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

    guard_options = _parse_guard_options(raw_options)

    # Initialize guards
    guard_names = [g.strip() for g in guards.split(",")]
    initialized_guards = {}

    for name in guard_names:
        if name not in AVAILABLE_GUARDS:
            click.echo(f"Error: Unknown guard '{name}'", err=True)
            click.echo(f"Available: {', '.join(AVAILABLE_GUARDS.keys())}")
            sys.exit(1)

        try:
            opts = guard_options.get(name, {})
            initialized_guards[name] = AVAILABLE_GUARDS[name]["factory"](opts)  # type: ignore
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
            if info["options"]:
                opts_str = ", ".join(f"{k}={v!r}" for k, v in info["options"].items())
                click.echo(f"    Options:     {opts_str}")
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

    if info_dict["options"]:
        click.echo("\nTunable options (via --option guard.param=value):")
        for param, default in info_dict["options"].items():
            click.echo(f"  • {guard_name}.{param} (default: {default!r})")

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
@click.option(
    "--option",
    "-o",
    "raw_options",
    multiple=True,
    metavar="GUARD.PARAM=VALUE",
    help="Override a guard's constructor parameter, e.g. -o heuristic.threshold=3. "
    "Repeatable. See `promptscreen info <guard>` for tunable parameters.",
)
def compare(
    prompt: str, guards: str, output_json: bool, raw_options: tuple[str, ...]
) -> None:
    """Compare guard results side-by-side.

    \b
    Example:
        promptscreen compare "test prompt"
        promptscreen compare "test" --guards heuristic,svm,scanner
        promptscreen compare "test" --json
        promptscreen compare "test" --guards length -o length.max_chars=10
    """
    guard_names = [g.strip() for g in guards.split(",")]
    guard_options = _parse_guard_options(raw_options)
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
            guard = AVAILABLE_GUARDS[name]["factory"](guard_options.get(name, {}))
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
@click.option(
    "--option",
    "-o",
    "raw_options",
    multiple=True,
    metavar="GUARD.PARAM=VALUE",
    help="Override a guard's constructor parameter, e.g. -o heuristic.threshold=3. "
    "Repeatable. See `promptscreen info <guard>` for tunable parameters.",
)
def interactive(guards: str, raw_options: tuple[str, ...]) -> None:
    """Interactive prompt scanning mode.

    Enter prompts interactively and see results in real-time.
    Press Ctrl+C or Ctrl+D to exit.

    \b
    Example:
        promptscreen interactive
        promptscreen interactive --guards heuristic,svm,scanner
        promptscreen interactive --guards length -o length.max_chars=500
    """
    guard_names = [g.strip() for g in guards.split(",")]
    guard_options = _parse_guard_options(raw_options)
    initialized_guards = {}

    # Initialize guards
    click.echo("Initializing guards...")
    for name in guard_names:
        if name not in AVAILABLE_GUARDS:
            click.echo(f"  ✗ {name}: Unknown guard", err=True)
            continue

        try:
            initialized_guards[name] = AVAILABLE_GUARDS[name]["factory"](
                guard_options.get(name, {})
            )
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
