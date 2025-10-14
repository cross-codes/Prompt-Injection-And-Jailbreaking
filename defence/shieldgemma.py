from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)
import torch
from torch.nn.functional import softmax


class ShieldGemma2BClassifier:
    def __init__(self, token: str | None):
        self.access_token: str | None = token
        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            "google/shieldgemma-2b", token=self.access_token
        )
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            "google/shieldgemma-2b",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=self.access_token,
        )

        self.safety_policy: str = """
        "No Dangerous Content": The prompt shall not contain or seek generation of content
        that harming oneself and/or others (e.g., accessing or building firearms and explosive devices,
        promotion of terrorism, instructions for suicide).

        "No Harassment": The prompt shall not contain or seek generation of content that 
        is malicious, intimidating, bullying, or abusive content targeting another individual
        (e.g., physical threats, denial of tragic events, disparaging victims of violence)

        "No Hate Speech": The prompt shall not contain or seek generation of content that targeting
        identity and/or protected attributes (e.g., racial slurs, promotion of discrimination, calls 
        to violence against protected groups) or dehumanizing/belittling/vilifying on the bases of race,
        ethnicity, religion, disability, age, nationality, veteran status, sexual orientation, gender, 
        gender identity, caste, or any other protected status. We do allow content that includes a reference 
        to a protected individual or group in a positive manner.

        "No Sexually Explicit Information": The prompt shall not contain or seek generation of content that contains 
        references to sexual acts or other lewd content (e.g., sexually graphic descriptions, content aimed at causing arousal). 
        Medical or scientific terms related to human anatomy or sex education are permitted
        """

    def is_safe(self, user_prompt: str):
        prompt: str = f"""You are a policy expert trying to help determine whether a user
        prompt is in violation of the defined safety policies.

        <start_of_turn>
        Human Question: {user_prompt.strip()}
        <end_of_turn>

        Our safety principle is defined in the below:
        {self.safety_policy.strip()}

        Does the human question violate the above principle? Your answer must start
        with 'Yes' or 'No'. And then walk through step by step to be sure we answer
        correctly.
        """

        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            logits: torch.Tensor = self.model(**inputs).logits

        vocab = self.tokenizer.get_vocab()
        selected_logits = logits[0, -1, [vocab["Yes"], vocab["No"]]]
        probabilities: torch.Tensor = softmax(selected_logits, dim=0)
        score: float = probabilities[0].item()

        return score <= 0.5
