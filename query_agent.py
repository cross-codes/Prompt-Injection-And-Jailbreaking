from langchain_ollama import OllamaLLM


class QueryAgent:
    def __init__(self, model: str, system_prompt: str = ""):
        self.llm: OllamaLLM = OllamaLLM(model=model)
        self.history_lst: list[tuple[str, str]] = []
        self.system_prompt: str = system_prompt

    def _generate_query(self, query: str, history: bool = False) -> str:
        prompt: str = ""
        if history:
            fmt_history: str = "\n".join(
                f"User: {q}\n Assistant: {a}" for q, a in self.history_lst
            )

            prompt = f"""
                --------------
                System prompt: {self.system_prompt},
                --------------
                Conversation History: {fmt_history},
                --------------
                User query: {query}
                --------------
            """

        else:
            prompt = f"""
                --------------
                System prompt: {self.system_prompt},
                --------------
                User query: {query}
                --------------
            """

        return prompt

    def query(self, query: str, history: bool = False) -> str:
        prompt: str = self._generate_query(query, history)
        response: str = self.llm.invoke(prompt)
        if history:
            self.history_lst.append((query, response))
        return response

    def clear_history(self):
        self.history_lst.clear()
