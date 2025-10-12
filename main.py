import json, time
from defence.classifier_cluster import ClassifierCluster
from metrics import AttackEvaluator, AttackResult, MetricsCalculator
from defence.heuristic_channel import HeuristicVectorAnalyzer
from query_agent import QueryAgent

if __name__ == "__main__":
    query_agent: QueryAgent = QueryAgent("tinyllama")
    cluster_guard: ClassifierCluster = ClassifierCluster()
    heuristic_guard: HeuristicVectorAnalyzer = HeuristicVectorAnalyzer(3, 3)

    evaluator: AttackEvaluator = AttackEvaluator("tinyllama", 0.1)
    metrics_calc: MetricsCalculator = MetricsCalculator()

    print("---------------------")
    print("Without GUARD enabled")
    print("---------------------")

    with open("prompts.json", "r") as fh:
        # fmt: off
        data = json.load(fh)  # pyright: ignore[reportAny]
        prompts: list[str] = data["prompts"] # pyright: ignore[reportAny, reportRedeclaration]
        for prompt in prompts:
            start_time: float = time.time() # pyright: ignore[reportRedeclaration]
            output: str = query_agent.query(prompt) # pyright: ignore[reportRedeclaration]
            response_time: float = time.time() - start_time # pyright: ignore[reportRedeclaration]
            attack_result: AttackResult = evaluator.evaluate(response_time, output, prompt) # pyright: ignore[reportUnknownMemberType, reportRedeclaration]

            metrics_calc.add_result(attack_result)
        # fmt: on

        metrics_calc.evaluate()

        print("---------------------")
        print("With GUARD enabled")
        print("---------------------")

        for prompt in prompts:
            if cluster_guard.is_safe(prompt) and heuristic_guard.is_safe(prompt):
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
                pass

        metrics_calc.evaluate()
