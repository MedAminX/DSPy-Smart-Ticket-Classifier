import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import dspy
from src.modules import TicketClassifierModule, get_ollama_lm
from src.signatures import ALLOWED_CATEGORIES, ALLOWED_PRIORITIES

DEFAULT_ARTIFACT_PATH = ROOT_DIR / "artifacts" / "optimized_ticket_classifier.json"
ORDER_ID_REGEX = re.compile(r"^ORD-\d{4}$")

_CACHED_PROGRAM: Optional[TicketClassifierModule] = None
_LM_CONFIGURED: bool = False


def get_compiled_classifier(
    artifact_path: Path = DEFAULT_ARTIFACT_PATH,
    model_name: str = "ollama_chat/qwen2.5:1.5b",
    api_base: str = "http://localhost:11434/v1",
) -> TicketClassifierModule:
    """Load and cache the compiled DSPy classifier once to save RAM in Streamlit."""
    global _CACHED_PROGRAM, _LM_CONFIGURED

    if not _LM_CONFIGURED:
        lm = get_ollama_lm(
            model_name=model_name,
            api_base=api_base,
            temperature=0.0,
            max_tokens=300,
        )
        dspy.configure(lm=lm)
        _LM_CONFIGURED = True

    if _CACHED_PROGRAM is None:
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Compiled DSPy artifact not found at: {artifact_path}. "
                "Run 'python -m src.optimize_dspy' first."
            )
        program = TicketClassifierModule(use_cot=True)
        program.load(str(artifact_path))
        _CACHED_PROGRAM = program

    return _CACHED_PROGRAM


def _sanitize_output(raw_category: Any, raw_order_id: Any, raw_priority: Any) -> Dict[str, Optional[str]]:
    """Enforce strict output types and allowed enum values before returning to Pydantic."""
    category = str(raw_category).strip() if raw_category is not None else "Other"
    if category not in ALLOWED_CATEGORIES:
        category = "Other"

    priority = str(raw_priority).strip() if raw_priority is not None else "Medium"
    if priority not in ALLOWED_PRIORITIES:
        priority = "Medium"

    order_id: Optional[str] = None
    if raw_order_id is not None:
        candidate = str(raw_order_id).strip().upper()
        if ORDER_ID_REGEX.match(candidate):
            order_id = candidate

    return {
        "category": category,
        "order_id": order_id,
        "priority": priority,
    }


def classify_ticket(ticket_text: str) -> Dict[str, Optional[str]]:
    """
    Public integration entrypoint for Student 2 (Streamlit + Pydantic).

    Args:
        ticket_text: Raw customer support ticket message.

    Returns:
        dict with keys:
            - category (str): One of 'Shipping', 'Refund', 'Payment', 'Technical', 'Other'
            - order_id (Optional[str]): Order ID matching 'ORD-XXXX' or None
            - priority (str): One of 'High', 'Medium', 'Low'
    """
    if not isinstance(ticket_text, str) or not ticket_text.strip():
        raise ValueError("ticket_text must be a non-empty string.")

    classifier = get_compiled_classifier()
    prediction = classifier(ticket_text=ticket_text.strip())

    return _sanitize_output(
        raw_category=getattr(prediction, "category", None),
        raw_order_id=getattr(prediction, "order_id", None),
        raw_priority=getattr(prediction, "priority", None),
    )


if __name__ == "__main__":
    samples = [
        "My package for order ORD-2045 has been delayed for 5 days, can you check?",
        "Your mobile app crashes with a 500 error every time I try to reset my password!",
    ]
    print("=== TESTING DSPy INTEGRATION ADAPTER ===")
    for idx, text in enumerate(samples, start=1):
        res = classify_ticket(text)
        print(f"\nSample {idx}: {text}")
        print(f"Output  : {res}")

