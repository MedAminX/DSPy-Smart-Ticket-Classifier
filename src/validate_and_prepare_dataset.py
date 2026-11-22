import json
import re
from collections import Counter
from pathlib import Path

ALLOWED_CATEGORIES = {"Shipping", "Refund", "Payment", "Technical", "Other"}
ALLOWED_PRIORITIES = {"High", "Medium", "Low"}
REQUIRED_FIELDS = {"id", "ticket_text", "category", "order_id", "priority"}
ORDER_ID_PATTERN = re.compile(r"^ORD-\d{4}$")
TEXT_ORDER_PATTERN = re.compile(r"\bORD-\d+\b")

def main():
    raw_path = Path("data/raw_tickets.json")
    clean_path = Path("data/clean_tickets.json")
    report_path = Path("data/dataset_quality_report.json")

    with raw_path.open("r", encoding="utf-8") as f:
        raw_tickets = json.load(f)

    issues = []
    seen_ids = set()
    seen_texts = set()
    clean_tickets = []

    for idx, ticket in enumerate(raw_tickets):
        # 1. Check required fields
        missing_fields = REQUIRED_FIELDS - set(ticket.keys())
        if missing_fields:
            issues.append(f"Row {idx}: Missing fields {missing_fields}")
            continue

        t_id = str(ticket["id"]).strip()
        text = str(ticket["ticket_text"]).strip()
        category = str(ticket["category"]).strip()
        priority = str(ticket["priority"]).strip()
        order_id = ticket["order_id"]

        # 2. Check non-empty text and valid labels
        if not t_id or not text:
            issues.append(f"Row {idx} ({t_id}): Empty id or ticket_text")
            continue
        if category not in ALLOWED_CATEGORIES:
            issues.append(f"Row {idx} ({t_id}): Invalid category '{category}'")
            continue
        if priority not in ALLOWED_PRIORITIES:
            issues.append(f"Row {idx} ({t_id}): Invalid priority '{priority}'")
            continue

        # 3. Check duplicates (ID and normalized text)
        norm_text = re.sub(r"\s+", " ", text.lower())
        if t_id in seen_ids:
            issues.append(f"Row {idx} ({t_id}): Duplicate ticket ID")
            continue
        if norm_text in seen_texts:
            issues.append(f"Row {idx} ({t_id}): Duplicate ticket_text")
            continue

        # 4. Validate order_id format and consistency with ticket_text
        if order_id is not None:
            order_id = str(order_id).strip()
            if not ORDER_ID_PATTERN.match(order_id):
                issues.append(f"Row {idx} ({t_id}): Invalid order_id format '{order_id}'")
                continue
            if order_id not in text:
                issues.append(f"Row {idx} ({t_id}): order_id '{order_id}' not found in ticket_text")
                continue
        else:
            found_in_text = TEXT_ORDER_PATTERN.findall(text)
            if found_in_text:
                issues.append(f"Row {idx} ({t_id}): order_id is None but text contains {found_in_text}")
                continue

        seen_ids.add(t_id)
        seen_texts.add(norm_text)
        clean_tickets.append({
            "id": t_id,
            "ticket_text": text,
            "category": category,
            "order_id": order_id,
            "priority": priority,
        })

    # 5. Audit balance and distributions
    cat_counts = dict(Counter(t["category"] for t in clean_tickets))
    prio_counts = dict(Counter(t["priority"] for t in clean_tickets))
    pair_counts = {
        f"{cat} | {prio}": sum(1 for t in clean_tickets if t["category"] == cat and t["priority"] == prio)
        for cat in sorted(ALLOWED_CATEGORIES)
        for prio in ["High", "Medium", "Low"]
    }
    with_order = sum(1 for t in clean_tickets if t["order_id"] is not None)
    without_order = sum(1 for t in clean_tickets if t["order_id"] is None)

    report = {
        "source_file": str(raw_path),
        "clean_file": str(clean_path),
        "total_raw_tickets": len(raw_tickets),
        "total_clean_tickets": len(clean_tickets),
        "issues_found": len(issues),
        "issues_details": issues,
        "category_distribution": cat_counts,
        "priority_distribution": prio_counts,
        "category_priority_matrix": pair_counts,
        "order_id_distribution": {
            "with_order_id": with_order,
            "without_order_id_null": without_order,
        },
    }

    with clean_path.open("w", encoding="utf-8") as f:
        json.dump(clean_tickets, f, indent=2, ensure_ascii=False)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=== DATASET QUALITY & BALANCE AUDIT ===")
    print(f"Raw tickets   : {len(raw_tickets)}")
    print(f"Clean tickets : {len(clean_tickets)}")
    print(f"Issues found  : {len(issues)}")
    print(f"Categories    : {cat_counts}")
    print(f"Priorities    : {prio_counts}")
    print(f"Order IDs     : with_order_id={with_order}, null={without_order}")
    print(f"Saved clean dataset to : {clean_path}")
    print(f"Saved audit report to  : {report_path}")

if __name__ == "__main__":
    main()

