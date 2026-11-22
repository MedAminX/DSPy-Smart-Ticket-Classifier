import unittest
from unittest.mock import patch

from pydantic import ValidationError
from src import schemas


class TicketRequestTests(unittest.TestCase):
    def test_valid_ticket(self):
        ticket = schemas.TicketRequest(
            ticket_text="  My order has not arrived.  "
        )
        self.assertEqual(
            ticket.ticket_text,
            "My order has not arrived.",
        )

    def test_empty_ticket(self):
        with self.assertRaises(ValidationError):
            schemas.TicketRequest(ticket_text="")

    def test_whitespace_only_ticket(self):
        with self.assertRaises(ValidationError):
            schemas.TicketRequest(ticket_text="   ")

    def test_invalid_input_type(self):
        with self.assertRaises(ValidationError):
            schemas.TicketRequest(ticket_text=123)

    def test_long_ticket(self):
        text = "A" * 10000
        ticket = schemas.TicketRequest(ticket_text=text)
        self.assertEqual(ticket.ticket_text, text)


class TicketResultTests(unittest.TestCase):
    def test_valid_result(self):
        result = schemas.TicketResult(
            category="Shipping",
            order_id="ORD-104",
            priority="Medium",
        )
        self.assertEqual(
            result.model_dump(),
            {
                "category": "Shipping",
                "order_id": "ORD-104",
                "priority": "Medium",
            },
        )

    def test_invalid_category(self):
        with self.assertRaises(ValidationError):
            schemas.TicketResult(
                category="Sales",
                priority="Medium",
            )

    def test_invalid_priority(self):
        with self.assertRaises(ValidationError):
            schemas.TicketResult(
                category="Shipping",
                priority="Urgent",
            )

    def test_missing_order_id(self):
        result = schemas.TicketResult(
            category="Shipping",
            priority="Medium",
        )
        self.assertIsNone(result.order_id)

    def test_null_order_id(self):
        result = schemas.TicketResult(
            category="Shipping",
            order_id=None,
            priority="Medium",
        )
        self.assertIsNone(result.order_id)

    def test_invalid_order_id_type(self):
        with self.assertRaises(ValidationError):
            schemas.TicketResult(
                category="Shipping",
                order_id=104,
                priority="Medium",
            )

    def test_missing_category(self):
        with self.assertRaises(ValidationError):
            schemas.TicketResult(priority="Medium")

    def test_missing_priority(self):
        with self.assertRaises(ValidationError):
            schemas.TicketResult(category="Shipping")


class StreamlitInterfaceTests(unittest.TestCase):
    def setUp(self):
        from pathlib import Path
        from streamlit.testing.v1 import AppTest

        app_path = Path(__file__).with_name("app.py").resolve()
        self.app = AppTest.from_file(
            str(app_path),
            default_timeout=15,
        ).run()

        self.assertEqual(len(self.app.exception), 0)

    def test_initial_page(self):
        self.assertEqual(
            self.app.title[0].value,
            "Smart Customer Support",
        )
        self.assertIn("Live DSPy Mode", self.app.info[0].value)
        self.assertEqual(self.app.text_area[0].value, "")
        self.assertEqual(self.app.button[0].label, "Classify")
        self.assertEqual(self.app.button[1].label, "Clear")

    def test_valid_ticket(self):
        sample_output = {
            "category": "Shipping",
            "order_id": "ORD-2045",
            "priority": "Medium",
        }
        self.app.text_area[0].input("My package for order ORD-2045 is delayed.")
        with patch("src.dspy_adapter.classify_ticket", return_value=sample_output):
            self.app.button[0].click().run()

        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.error), 0)
        self.assertEqual(self.app.subheader[0].value, "Classification Result")

        displayed_text = "\n".join(
            element.value for element in self.app.markdown
        )
        self.assertIn("Shipping", displayed_text)
        self.assertIn("ORD-2045", displayed_text)
        self.assertIn("Medium", displayed_text)

    def test_clear_button(self):
        sample_output = {
            "category": "Other",
            "order_id": None,
            "priority": "Low",
        }
        self.app.text_area[0].input("My order has not arrived.")
        with patch("src.dspy_adapter.classify_ticket", return_value=sample_output):
            self.app.button[0].click().run()
        self.assertEqual(self.app.subheader[0].value, "Classification Result")

        self.app.button[1].click().run()

        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(self.app.text_area[0].value, "")
        self.assertEqual(len(self.app.subheader), 0)
        self.assertEqual(len(self.app.error), 0)

    def test_empty_ticket(self):
        self.app.button[0].click().run()

        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.error), 1)
        self.assertEqual(
            self.app.error[0].value,
            "Please enter a non-empty customer ticket.",
        )
        self.assertEqual(len(self.app.subheader), 0)

    def test_whitespace_only_ticket(self):
        self.app.text_area[0].input("   ")
        self.app.button[0].click().run()

        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.error), 1)
        self.assertEqual(
            self.app.error[0].value,
            "Please enter a non-empty customer ticket.",
        )
        self.assertEqual(len(self.app.subheader), 0)

    def test_long_ticket(self):
        text = "A" * 10000
        sample_output = {
            "category": "Other",
            "order_id": None,
            "priority": "Low",
        }
        self.app.text_area[0].input(text)
        with patch("src.dspy_adapter.classify_ticket", return_value=sample_output):
            self.app.button[0].click().run()

        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.error), 0)
        self.assertEqual(self.app.text_area[0].value, text)
        self.assertEqual(self.app.subheader[0].value, "Classification Result")

    def test_invalid_classifier_output(self):
        invalid_result = {
            "category": "Sales",
            "order_id": None,
            "priority": "Low",
        }
        self.app.text_area[0].input("I need help with my order.")

        with patch(
            "src.dspy_adapter.classify_ticket",
            return_value=invalid_result,
        ):
            with self.assertLogs(level="ERROR"):
                self.app.button[0].click().run()

        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.error), 1)
        self.assertEqual(
            self.app.error[0].value,
            "The classifier returned invalid data.",
        )
        self.assertEqual(len(self.app.subheader), 0)

    def test_classifier_failure_and_recovery(self):
        valid_recovery = {
            "category": "Shipping",
            "order_id": None,
            "priority": "Medium",
        }
        self.app.text_area[0].input("I need help with my order.")

        with patch(
            "src.dspy_adapter.classify_ticket",
            side_effect=[RuntimeError("Simulated classifier failure"), valid_recovery],
        ):
            with self.assertLogs(level="ERROR"):
                self.app.button[0].click().run()

            self.assertEqual(len(self.app.exception), 0)
            self.assertEqual(len(self.app.error), 1)
            self.assertEqual(
                self.app.error[0].value,
                "Classification failed. Please try again.",
            )
            self.assertEqual(len(self.app.subheader), 0)

            self.app.button[0].click().run()

        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.error), 0)
        self.assertEqual(self.app.subheader[0].value, "Classification Result")
