from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TicketRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        strict=True,
    )

    ticket_text: str = Field(min_length=1)


class TicketResult(BaseModel):
    model_config = ConfigDict(strict=True)

    category: Literal["Shipping", "Refund", "Payment", "Technical", "Other"]
    order_id: str | None = None
    priority: Literal["High", "Medium", "Low"]
