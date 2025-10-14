import json, time, os
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

    cnt = 0
    with open("offence/advllm/prompts.json", "r") as fh:
        # fmt: off
        data = json.load(fh)  # pyright: ignore[reportAny]
        prompts: list[str] = data["prompts"] # pyright: ignore[reportAny, reportRedeclaration]
        # fmt: on

        for prompt in prompts:
            if (
                cluster_guard.is_safe(prompt)
                and heuristic_guard.is_safe(prompt)
                and llm_guard.is_safe(prompt)
            ):
                start_time: float = time.time()
                output: str = query_agent.query(prompt)
                response_time: float = time.time() - start_time
                # fmt: off
                attack_result: AttackResult = evaluator.evaluate(response_time, output, prompt) # pyright: ignore[reportUnknownMemberType]
                # fmt: on

                metrics_calc.add_result(attack_result)
            else:
                failed_result: AttackResult = evaluator.create_failed_attempt(prompt)
                metrics_calc.add_result(failed_result)
            print(f'Prompt {cnt} assesed')
            cnt += 1

    with open("offence/emoji/prompts.json", "r") as fh:
        # fmt: off
        data = json.load(fh)  # pyright: ignore[reportAny]
        prompts: list[str] = data["prompts"] # pyright: ignore[reportAny, reportRedeclaration]
        # fmt: on

        for prompt in prompts:
            if (
                cluster_guard.is_safe(prompt)
                and heuristic_guard.is_safe(prompt)
                and llm_guard.is_safe(prompt)
            ):
                start_time: float = time.time()
                output: str = query_agent.query(prompt)
                response_time: float = time.time() - start_time
                # fmt: off
                attack_result: AttackResult = evaluator.evaluate(response_time, output, prompt) # pyright: ignore[reportUnknownMemberType]
                # fmt: on

                metrics_calc.add_result(attack_result)
            else:
                failed_result: AttackResult = evaluator.create_failed_attempt(prompt)
                metrics_calc.add_result(failed_result)
            print(f'Prompt {cnt} assesed')
            cnt += 1

    with open("offence/regular/prompts.json", "r") as fh:
        # fmt: off
        data = json.load(fh)  # pyright: ignore[reportAny]
        prompts: list[str] = data["prompts"] # pyright: ignore[reportAny, reportRedeclaration]
        # fmt: on

        for prompt in prompts:
            if (
                cluster_guard.is_safe(prompt)
                and heuristic_guard.is_safe(prompt)
                and llm_guard.is_safe(prompt)
            ):
                start_time: float = time.time()
                output: str = query_agent.query(prompt)
                response_time: float = time.time() - start_time
                # fmt: off
                attack_result: AttackResult = evaluator.evaluate(response_time, output, prompt) # pyright: ignore[reportUnknownMemberType]
                # fmt: on

                metrics_calc.add_result(attack_result)
            else:
                failed_result: AttackResult = evaluator.create_failed_attempt(prompt)
                metrics_calc.add_result(failed_result)
            print(f'Prompt {cnt} assesed')
            cnt += 1

    metrics_calc.evaluate()
