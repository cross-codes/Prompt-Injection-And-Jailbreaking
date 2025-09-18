import json, time
from defence.classifier_cluster import ClassifierCluster
from query_agent import QueryAgent

if __name__ == "__main__":
    query_agent: QueryAgent = QueryAgent("tinyllama")
    guard: ClassifierCluster = ClassifierCluster()
    # evaluator = AttackEvaluator(name of the model, temperature)
    # metrics_calc = MetricsCalculator()
    # Metrics calculator should have a list of attack results as a member variable

    with open("prompts.json", "r") as fh:
        data = json.load(fh)  # pyright: ignore[reportExplicitAny]
        prompts: list[str] = data["prompts"]
        for prompt in prompts:
            # (1) Pass it through the cluster
            # (2) If cluster fails: add a failed attack to list
            # (3) If success: evaluate the attack.
            if (guard.is_safe(prompt)):
                start_time = time.time();
                output: str = query_agent.query(prompt)
                response_time = time.time() - start_time

                # Create a new AttackResult object
                # I expect the evaluator to result an attackresult
                # attack_result = evalutor.evaluate(response_time, output, prompt, ...)
                # Maybe ask LLM to generate the rest of the fields in the AttackResult class

                # metircs_calc.append_result(attack_result) -> Add to the list member variables
            else:
                pass
                # Create a failed attack_result
                # Create a method: failed_result = evaluator.create_failed_attempt(prompt) -> AttackResult
                # Alter: failed_attempt: AttackResult(success = false)
                # metrics_calc.append_resuilt(failed_result)

        #metrics_calc.evaluate() -> Show final result after all prompts
