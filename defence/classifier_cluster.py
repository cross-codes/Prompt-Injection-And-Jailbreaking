import torch
from torch.types import Number, Tensor
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BatchEncoding,
    PreTrainedTokenizerFast,
    PreTrainedModel,
)


class ClassifierCluster:
    def __init__(self):
        # fmt: off
        self.TOXICITY_MODEL_NAME: str = "textdetox/xlmr-large-toxicity-classifier-v2"
        self.tox_tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
            self.TOXICITY_MODEL_NAME
        )
        self.tox_model: PreTrainedModel = (
            AutoModelForSequenceClassification.from_pretrained(self.TOXICITY_MODEL_NAME)  # pyright: ignore[reportUnknownMemberType]
        )
        self.tox_mx_tokens: int = self.tox_tokenizer.model_max_length  # pyright: ignore[reportUnknownMemberType]

        self.JAILBREAK_MODEL_NAME: str = "jackhhao/jailbreak-classifier"
        self.jbreak_tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
            self.JAILBREAK_MODEL_NAME
        )
        self.jbreak_model: PreTrainedModel = (
            AutoModelForSequenceClassification.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
                self.JAILBREAK_MODEL_NAME
            )
        )
        self.jbreak_mx_tokens: int = self.jbreak_tokenizer.model_max_length  # pyright: ignore[reportUnknownMemberType]
        # fmt: on

    def _is_jailbreak(self, query: str) -> bool:
        inputs: BatchEncoding = self.jbreak_tokenizer(
            query,
            max_length=self.jbreak_mx_tokens,
            truncation=True,
            return_overflowing_tokens=True,
            stride=50,
            padding = "max_length",
            return_tensors="pt",
        )

        for i in range(len(inputs["input_ids"])):  # pyright: ignore[reportArgumentType]
            # fmt: off
            chunk_ids: Tensor = inputs["input_ids"][i] # pyright: ignore[reportIndexIssue, reportUnknownVariableType]
            chunk_attention_mask: Tensor = inputs["attention_mask"][i]  # pyright: ignore[reportIndexIssue, reportUnknownVariableType]

            with torch.no_grad():
                output = self.jbreak_model(  # pyright: ignore[reportAny]
                    input_ids=chunk_ids.unsqueeze(0),  # pyright: ignore[reportUnknownMemberType]
                    attention_mask=chunk_attention_mask.unsqueeze(0),  # pyright: ignore[reportUnknownMemberType]
                )

            logits: Tensor = output.logits  # pyright: ignore[reportAny]
            predicted_class_id: Number = torch.argmax(logits, dim=1).item()
            if predicted_class_id == 1:
                return True
            # fmt: on

        return False

    def _is_toxic(self, query: str) -> bool:
        inputs: BatchEncoding = self.tox_tokenizer(
            query,
            max_length=self.tox_mx_tokens,
            truncation=True,
            padding = "max_length",
            return_overflowing_tokens=True,
            stride=50,
            return_tensors="pt",
        )

        for i in range(len(inputs["input_ids"])):  # pyright: ignore[reportArgumentType]
            # fmt: off
            chunk_ids: Tensor = inputs["input_ids"][i] # pyright: ignore[reportIndexIssue, reportUnknownVariableType]
            chunk_attention_mask: Tensor = inputs["attention_mask"][i]  # pyright: ignore[reportIndexIssue, reportUnknownVariableType]

            with torch.no_grad():
                output = self.tox_model(  # pyright: ignore[reportAny]
                    input_ids=chunk_ids.unsqueeze(0),  # pyright: ignore[reportUnknownMemberType]
                    attention_mask=chunk_attention_mask.unsqueeze(0),  # pyright: ignore[reportUnknownMemberType]
                )

            logits: Tensor = output.logits  # pyright: ignore[reportAny]
            predicted_class_id: Number = torch.argmax(logits, dim=1).item()
            if predicted_class_id == 1:
                return True
            # fmt: on

        return False

    def is_safe(self, query: str) -> bool:
        return not (self._is_jailbreak(query) or self._is_toxic(query))
