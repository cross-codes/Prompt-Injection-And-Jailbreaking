import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class OutputScanner:
    def __init__(self) -> None:
        self.rejection_model_name = "protectai/distilroberta-base-rejection-v1"
        self.bias_model_name = "valurank/distilroberta-bias"

        self.rejection_tokenizer = AutoTokenizer.from_pretrained(
            self.rejection_model_name
        )
        self.rejection_model = AutoModelForSequenceClassification.from_pretrained(
            self.rejection_model_name
        )

        self.bias_tokenizer = AutoTokenizer.from_pretrained(self.bias_model_name)
        self.bias_model = AutoModelForSequenceClassification.from_pretrained(
            self.bias_model_name
        )

        self.max_tokens_rejection = self.rejection_tokenizer.model_max_length
        self.max_tokens_bias = self.bias_tokenizer.model_max_length

    def scan_output(self, text: str) -> str:
        rej_inputs = self.rejection_tokenizer(
            text,
            max_length=self.max_tokens_rejection,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        with torch.no_grad():
            rej_output = self.rejection_model(**rej_inputs)
        rej_logits = rej_output.logits
        rej_pred = torch.argmax(rej_logits, dim=1).item()

        # protectai/distilroberta-base-rejection-v1 id2label: 0=NORMAL, 1=REJECTION
        if rej_pred == 1:
            return "Warning: Output rejected due to unsafe or disallowed content."

        bias_inputs = self.bias_tokenizer(
            text,
            max_length=self.max_tokens_bias,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        with torch.no_grad():
            bias_output = self.bias_model(**bias_inputs)
        bias_logits = bias_output.logits
        bias_pred = torch.argmax(bias_logits, dim=1).item()

        # valurank/distilroberta-bias id2label: 0=BIASED, 1=NEUTRAL (inverted
        # relative to the rejection model above -- verified against the
        # model's config.json, this is intentional, not an off-by-one bug)
        if bias_pred == 0:
            return "Warning: Output detected with potential bias."

        return text
