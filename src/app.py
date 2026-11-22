import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from pydantic import ValidationError

from src.dspy_adapter import classify_ticket
from src.schemas import TicketRequest, TicketResult


def clear_ticket() -> None:
    st.session_state["ticket_text"] = ""


st.set_page_config(
    page_title="Smart Customer Support",
    layout="centered",
)

st.title("Smart Customer Support")
st.info("Live DSPy Mode — Compiled Chain-of-Thought Classifier (Qwen 2.5 1.5B via Ollama)")

ticket_text = st.text_area(
    "Customer ticket",
    height=180,
    key="ticket_text",
)

classify_column, clear_column = st.columns(2)

classify_clicked = classify_column.button(
    "Classify",
    type="primary",
)

clear_column.button("Clear", on_click=clear_ticket)

if classify_clicked:
    try:
        request = TicketRequest(ticket_text=ticket_text)
    except ValidationError:
        st.error("Please enter a non-empty customer ticket.")
        st.stop()

    try:
        with st.spinner("Classifying ticket with compiled DSPy pipeline..."):
            raw_result = classify_ticket(request.ticket_text)
        payload = raw_result.model_dump() if hasattr(raw_result, "model_dump") else raw_result
        result = TicketResult.model_validate(payload)
    except ValidationError:
        logging.exception("Classifier output validation failed")
        st.error("The classifier returned invalid data.")
    except Exception:
        logging.exception("Classification failed")
        st.error("Classification failed. Please try again.")
    else:
        st.subheader("Classification Result")
        st.write("**Category**", result.category)
        st.write(
            "**Order ID**",
            result.order_id if result.order_id is not None else "Not provided",
        )
        st.write("**Priority**", result.priority)
