from typing import Literal, Optional
import dspy

ALLOWED_CATEGORIES = ("Shipping", "Refund", "Payment", "Technical", "Other")
ALLOWED_PRIORITIES = ("High", "Medium", "Low")

CategoryLiteral = Literal["Shipping", "Refund", "Payment", "Technical", "Other"]
PriorityLiteral = Literal["High", "Medium", "Low"]


class TicketClassificationSignature(dspy.Signature):
    """Classify a customer support message into the exact department category, extract any explicit order ID, and assign its urgency priority."""

    ticket_text: str = dspy.InputField(
        desc="The customer support ticket message describing a question, request, or issue."
    )

    category: CategoryLiteral = dspy.OutputField(
        desc=(
            "Choose strictly one department based on the core topic: "
            "'Shipping' (package delivery, courier, tracking status, customs, shipping address/fees), "
            "'Refund' (returning an item, money back, reimbursement, order cancellation for refund), "
            "'Payment' (credit/debit card charges, billing, declined payment, invoice, promo code, payment methods), "
            "'Technical' (website/app bugs, login/2FA errors, server 500/404 errors, software license/firmware/UI issues), "
            "or 'Other' (account ownership/merging, GDPR/legal, wholesale partnership, staff complaint, general hours/feedback)."
        )
    )
    order_id: Optional[str] = dspy.OutputField(
        desc="Extract the exact order ID matching 'ORD-XXXX' ONLY if explicitly written in ticket_text; otherwise return null."
    )
    priority: PriorityLiteral = dspy.OutputField(
        desc=(
            "Choose strictly one urgency level: "
            "'High' (urgent deadline, outage/locked out, double/unauthorized charge, lost/damaged delivery, security/safety/legal), "
            "'Medium' (standard issue needing action: declined card, return label, delayed tracking, address typo, bug workaround), "
            "or 'Low' (general policy question, opening hours, feedback/thank you, non-urgent inquiry)."
        )
    )

