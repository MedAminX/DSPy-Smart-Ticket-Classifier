import os
from typing import Optional
import dspy
from src.signatures import (
    ALLOWED_CATEGORIES,
    ALLOWED_PRIORITIES,
    TicketClassificationSignature,
)


def get_ollama_lm(
    model_name: str = "ollama_chat/qwen2.5:1.5b",
    api_base: str = "http://localhost:11434/v1",
    temperature: float = 0.0,
    max_tokens: int = 300,
) -> dspy.LM:
    """Configure a lightweight local Ollama LM instance for DSPy 3.5.0."""
    os.environ.setdefault("OLLAMA_API_KEY", "ollama")
    return dspy.LM(
        model=model_name,
        api_base=api_base,
        api_key="ollama",
        temperature=temperature,
        max_tokens=max_tokens,
    )


class TicketClassifierModule(dspy.Module):
    """Unified DSPy Module for Ticket Classification, Order ID Extraction, and Priority Prediction."""

    def __init__(self, use_cot: bool = True):
        super().__init__()
        self.use_cot = use_cot
        if use_cot:
            self.predictor = dspy.ChainOfThought(TicketClassificationSignature)
        else:
            self.predictor = dspy.Predict(TicketClassificationSignature)

    @staticmethod
    def _clean_category(value: object) -> str:
        if value is None:
            return "Invalid"
        text = str(value).strip().strip("'\"").capitalize()
        return text if text in ALLOWED_CATEGORIES else str(value).strip()

    @staticmethod
    def _clean_order_id(value: object) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip().strip("'\"")
        if text.lower() in {"", "null", "none", "n/a", "na", "unknown", "not mentioned"}:
            return None
        return text.upper()

    @staticmethod
    def _clean_priority(value: object) -> str:
        if value is None:
            return "Invalid"
        text = str(value).strip().strip("'\"").capitalize()
        return text if text in ALLOWED_PRIORITIES else str(value).strip()

    def forward(self, ticket_text: str) -> dspy.Prediction:
        raw_pred = self.predictor(ticket_text=ticket_text)

        cleaned_category = self._clean_category(getattr(raw_pred, "category", None))
        cleaned_order_id = self._clean_order_id(getattr(raw_pred, "order_id", None))
        cleaned_priority = self._clean_priority(getattr(raw_pred, "priority", None))
        reasoning = getattr(raw_pred, "reasoning", None)

        return dspy.Prediction(
            category=cleaned_category,
            order_id=cleaned_order_id,
            priority=cleaned_priority,
            reasoning=reasoning,
        )

