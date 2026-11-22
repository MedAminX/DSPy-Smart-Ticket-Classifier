from src.schemas import TicketRequest, TicketResult


def classify_ticket(ticket_text: str) -> TicketResult:
    """Return fixed mock data for interface testing."""
    TicketRequest(ticket_text=ticket_text)

    return TicketResult(
        category="Other",
        order_id=None,
        priority="Low",
    )
