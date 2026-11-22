import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import dspy
from src.metrics import evaluate_predictions
from src.modules import TicketClassifierModule, get_ollama_lm


def evaluate_optimized_on_test(module: dspy.Module, test_path: Path) -> Dict[str, Any]:
    with test_path.open("r", encoding="utf-8") as f:
        tickets = json.load(f)

    records: List[Dict[str, Any]] = []
    start_time = time.perf_counter()

    print(f"\n--- Evaluating OPTIMIZED DSPy on TEST ({len(tickets)} tickets) ---")
    for idx, item in enumerate(tickets, start=1):
        t0 = time.perf_counter()
        try:
            pred = module(ticket_text=item["ticket_text"])
            pred_cat, pred_ord, pred_prio = pred.category, pred.order_id, pred.priority
            err = None
        except Exception as exc:
            pred_cat, pred_ord, pred_prio = "Error", None, "Error"
            err = str(exc)

        elapsed = time.perf_counter() - t0
        records.append({
            "id": item["id"],
            "ticket_text": item["ticket_text"],
            "true_category": item["category"],
            "pred_category": pred_cat,
            "true_order_id": item["order_id"],
            "pred_order_id": pred_ord,
            "true_priority": item["priority"],
            "pred_priority": pred_prio,
            "latency_sec": round(elapsed, 3),
            "error": err,
        })
        print(
            f"[{idx:02d}/{len(tickets):02d}] {item['id']} | "
            f"Cat: {pred_cat} (true: {item['category']}) | "
            f"Ord: {pred_ord} (true: {item['order_id']}) | "
            f"Prio: {pred_prio} (true: {item['priority']}) | {elapsed:.2f}s"
        )

    total_runtime = time.perf_counter() - start_time
    metrics = evaluate_predictions(records)
    metrics["runtime_sec"] = round(total_runtime, 3)
    metrics["avg_latency_per_ticket_sec"] = round(total_runtime / len(tickets), 3) if tickets else 0.0

    return {
        "metrics": metrics,
        "predictions": records,
    }


def fmt_delta(before: float, after: float) -> str:
    diff = after - before
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.4f}"


def main() -> None:
    # Use the exact same LM evaluation settings as Baseline (temperature=0.0, max_tokens=300)
    lm = get_ollama_lm(temperature=0.0, max_tokens=300)
    dspy.configure(lm=lm)

    # 1. Load Baseline Test results (recorded earlier in Phase 6)
    baseline_path = Path("results/baseline_results.json")
    with baseline_path.open("r", encoding="utf-8") as f:
        baseline_data = json.load(f)
    base_m = baseline_data["test"]["metrics"]

    # 2. Load Optimization metadata and Compiled DSPy Artifact
    opt_report_path = Path("results/optimization_report.json")
    with opt_report_path.open("r", encoding="utf-8") as f:
        opt_report = json.load(f)
    selected_candidate = opt_report.get("selected_best_candidate", "ClassBalanced_FewShot_5")

    artifact_path = Path("artifacts/optimized_ticket_classifier.json")
    optimized_module = TicketClassifierModule(use_cot=True)
    optimized_module.load(str(artifact_path))

    inner_predict = getattr(optimized_module.predictor, "predict", optimized_module.predictor)
    loaded_demos = len(getattr(inner_predict, "demos", []))
    print(f"Loaded artifact: {artifact_path} ({selected_candidate}) | Active demos: {loaded_demos}")

    # 3. Evaluate Optimized DSPy on the isolated Test Set
    opt_eval = evaluate_optimized_on_test(optimized_module, Path("data/test.json"))
    opt_m = opt_eval["metrics"]

    comparison_table = {
        "exact_match_accuracy": {
            "baseline": base_m["exact_match_accuracy"],
            "optimized": opt_m["exact_match_accuracy"],
            "delta": round(opt_m["exact_match_accuracy"] - base_m["exact_match_accuracy"], 4),
        },
        "category_accuracy": {
            "baseline": base_m["category_metrics"]["accuracy"],
            "optimized": opt_m["category_metrics"]["accuracy"],
            "delta": round(opt_m["category_metrics"]["accuracy"] - base_m["category_metrics"]["accuracy"], 4),
        },
        "category_macro_precision": {
            "baseline": base_m["category_metrics"]["macro_precision"],
            "optimized": opt_m["category_metrics"]["macro_precision"],
            "delta": round(opt_m["category_metrics"]["macro_precision"] - base_m["category_metrics"]["macro_precision"], 4),
        },
        "category_macro_recall": {
            "baseline": base_m["category_metrics"]["macro_recall"],
            "optimized": opt_m["category_metrics"]["macro_recall"],
            "delta": round(opt_m["category_metrics"]["macro_recall"] - base_m["category_metrics"]["macro_recall"], 4),
        },
        "category_macro_f1": {
            "baseline": base_m["category_metrics"]["macro_f1"],
            "optimized": opt_m["category_metrics"]["macro_f1"],
            "delta": round(opt_m["category_metrics"]["macro_f1"] - base_m["category_metrics"]["macro_f1"], 4),
        },
        "priority_accuracy": {
            "baseline": base_m["priority_metrics"]["accuracy"],
            "optimized": opt_m["priority_metrics"]["accuracy"],
            "delta": round(opt_m["priority_metrics"]["accuracy"] - base_m["priority_metrics"]["accuracy"], 4),
        },
        "priority_macro_precision": {
            "baseline": base_m["priority_metrics"]["macro_precision"],
            "optimized": opt_m["priority_metrics"]["macro_precision"],
            "delta": round(opt_m["priority_metrics"]["macro_precision"] - base_m["priority_metrics"]["macro_precision"], 4),
        },
        "priority_macro_recall": {
            "baseline": base_m["priority_metrics"]["macro_recall"],
            "optimized": opt_m["priority_metrics"]["macro_recall"],
            "delta": round(opt_m["priority_metrics"]["macro_recall"] - base_m["priority_metrics"]["macro_recall"], 4),
        },
        "priority_macro_f1": {
            "baseline": base_m["priority_metrics"]["macro_f1"],
            "optimized": opt_m["priority_metrics"]["macro_f1"],
            "delta": round(opt_m["priority_metrics"]["macro_f1"] - base_m["priority_metrics"]["macro_f1"], 4),
        },
        "order_id_present_only_accuracy": {
            "baseline": base_m["order_id_metrics"]["present_only_accuracy"],
            "optimized": opt_m["order_id_metrics"]["present_only_accuracy"],
            "delta": round(opt_m["order_id_metrics"]["present_only_accuracy"] - base_m["order_id_metrics"]["present_only_accuracy"], 4),
        },
        "order_id_null_specificity": {
            "baseline": base_m["order_id_metrics"]["null_only_specificity"],
            "optimized": opt_m["order_id_metrics"]["null_only_specificity"],
            "delta": round(opt_m["order_id_metrics"]["null_only_specificity"] - base_m["order_id_metrics"]["null_only_specificity"], 4),
        },
        "order_id_entity_precision": {
            "baseline": base_m["order_id_metrics"]["entity_precision"],
            "optimized": opt_m["order_id_metrics"]["entity_precision"],
            "delta": round(opt_m["order_id_metrics"]["entity_precision"] - base_m["order_id_metrics"]["entity_precision"], 4),
        },
        "order_id_entity_recall": {
            "baseline": base_m["order_id_metrics"]["entity_recall"],
            "optimized": opt_m["order_id_metrics"]["entity_recall"],
            "delta": round(opt_m["order_id_metrics"]["entity_recall"] - base_m["order_id_metrics"]["entity_recall"], 4),
        },
        "order_id_entity_f1": {
            "baseline": base_m["order_id_metrics"]["entity_f1"],
            "optimized": opt_m["order_id_metrics"]["entity_f1"],
            "delta": round(opt_m["order_id_metrics"]["entity_f1"] - base_m["order_id_metrics"]["entity_f1"], 4),
        },
        "total_runtime_sec": {
            "baseline": base_m["runtime_sec"],
            "optimized": opt_m["runtime_sec"],
            "delta": round(opt_m["runtime_sec"] - base_m["runtime_sec"], 3),
        },
        "avg_latency_per_ticket_sec": {
            "baseline": base_m["avg_latency_per_ticket_sec"],
            "optimized": opt_m["avg_latency_per_ticket_sec"],
            "delta": round(opt_m["avg_latency_per_ticket_sec"] - base_m["avg_latency_per_ticket_sec"], 3),
        },
    }

    report = {
        "evaluation_dataset": "data/test.json",
        "test_samples": 15,
        "model": "ollama_chat/qwen2.5:1.5b",
        "selected_candidate": selected_candidate,
        "loaded_demos_count": loaded_demos,
        "side_by_side_comparison": comparison_table,
        "baseline_test_metrics": base_m,
        "optimized_test_metrics": opt_m,
        "optimized_test_predictions": opt_eval["predictions"],
    }

    out_path = Path("results/before_after_comparison.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n================ BEFORE vs AFTER COMPARISON (TEST SET) ================")
    print(f"{'Metric':<32} | {'Baseline':>10} | {'Optimized':>10} | {'Delta':>10}")
    print("-" * 70)
    for k, v in comparison_table.items():
        print(f"{k:<32} | {v['baseline']:>10} | {v['optimized']:>10} | {fmt_delta(v['baseline'], v['optimized']):>10}")
    print(f"\nSaved full comparison report to: {out_path}")


if __name__ == "__main__":
    main()

