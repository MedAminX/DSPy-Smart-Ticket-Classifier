from typing import Any, Dict, List, Optional, Sequence
import dspy
from src.signatures import ALLOWED_CATEGORIES, ALLOWED_PRIORITIES


def normalize_order_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().strip("'\"")
    if text.lower() in {"", "null", "none", "n/a", "na", "unknown", "not mentioned"}:
        return None
    return text.upper()


def compute_multiclass_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> Dict[str, Any]:
    total = len(y_true)
    if total == 0:
        return {"accuracy": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0, "per_class": {}}

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / total

    per_class: Dict[str, Dict[str, float]] = {}
    precisions, recalls, f1s = [], [], []

    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class[label] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": sum(1 for t in y_true if t == label),
        }
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    n_labels = len(labels)
    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(sum(precisions) / n_labels, 4),
        "macro_recall": round(sum(recalls) / n_labels, 4),
        "macro_f1": round(sum(f1s) / n_labels, 4),
        "per_class": per_class,
    }


def compute_order_id_metrics(
    y_true: Sequence[Optional[str]],
    y_pred: Sequence[Optional[str]],
) -> Dict[str, Any]:
    total = len(y_true)
    if total == 0:
        return {}

    overall_acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / total

    present_pairs = [(t, p) for t, p in zip(y_true, y_pred) if t is not None]
    present_matches = sum(1 for t, p in present_pairs if t == p)
    present_acc = present_matches / len(present_pairs) if present_pairs else 0.0

    null_pairs = [(t, p) for t, p in zip(y_true, y_pred) if t is None]
    null_matches = sum(1 for _, p in null_pairs if p is None)
    null_specificity = null_matches / len(null_pairs) if null_pairs else 0.0

    tp = sum(1 for t, p in zip(y_true, y_pred) if t is not None and p == t)
    fp = sum(1 for t, p in zip(y_true, y_pred) if p is not None and p != t)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t is not None and p != t)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "overall_accuracy": round(overall_acc, 4),
        "present_only_count": len(present_pairs),
        "present_only_exact_matches": present_matches,
        "present_only_accuracy": round(present_acc, 4),
        "null_only_count": len(null_pairs),
        "null_only_specificity": round(null_specificity, 4),
        "entity_precision": round(precision, 4),
        "entity_recall": round(recall, 4),
        "entity_f1": round(f1, 4),
    }


def evaluate_predictions(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute separated field-level metrics and full-record Exact Match Accuracy."""
    total = len(records)
    cat_true = [r["true_category"] for r in records]
    cat_pred = [r["pred_category"] for r in records]

    prio_true = [r["true_priority"] for r in records]
    prio_pred = [r["pred_priority"] for r in records]

    ord_true = [normalize_order_id(r["true_order_id"]) for r in records]
    ord_pred = [normalize_order_id(r["pred_order_id"]) for r in records]

    exact_matches = sum(
        1
        for ct, cp, pt, pp, ot, op in zip(cat_true, cat_pred, prio_true, prio_pred, ord_true, ord_pred)
        if ct == cp and pt == pp and ot == op
    )

    return {
        "total_samples": total,
        "exact_match_count": exact_matches,
        "exact_match_accuracy": round(exact_matches / total, 4) if total > 0 else 0.0,
        "category_metrics": compute_multiclass_metrics(cat_true, cat_pred, ALLOWED_CATEGORIES),
        "priority_metrics": compute_multiclass_metrics(prio_true, prio_pred, ALLOWED_PRIORITIES),
        "order_id_metrics": compute_order_id_metrics(ord_true, ord_pred),
    }


def dspy_ticket_metric(example: dspy.Example, pred: dspy.Prediction, trace: Any = None) -> float:
    """Weighted composite reward metric for DSPy optimization (0.0 to 1.0)."""
    cat_ok = 1.0 if str(getattr(pred, "category", "")).strip() == str(example.category).strip() else 0.0
    prio_ok = 1.0 if str(getattr(pred, "priority", "")).strip() == str(example.priority).strip() else 0.0
    ord_ok = 1.0 if normalize_order_id(getattr(pred, "order_id", None)) == normalize_order_id(example.order_id) else 0.0

    score = (0.40 * cat_ok) + (0.35 * prio_ok) + (0.25 * ord_ok)
    if trace is not None:
        return 1.0 if (cat_ok == 1.0 and prio_ok == 1.0 and ord_ok == 1.0) else 0.0
    return score

