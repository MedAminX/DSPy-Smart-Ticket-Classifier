import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import dspy
from src.metrics import evaluate_predictions
from src.modules import TicketClassifierModule, get_ollama_lm


def check_ollama_server(api_base: str = "http://localhost:11434") -> None:
    try:
        with urllib.request.urlopen(f"{api_base}/api/tags", timeout=5) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Ollama responded with HTTP {resp.status}")
    except Exception as exc:
        raise RuntimeError(
            "Cannot connect to Ollama at http://localhost:11434. "
            "Please make sure Ollama is running before executing this script."
        ) from exc


def evaluate_split(
    module: dspy.Module,
    dataset_path: Path,
    split_name: str,
) -> Dict[str, Any]:
    with dataset_path.open("r", encoding="utf-8") as f:
        tickets = json.load(f)

    records: List[Dict[str, Any]] = []
    start_time = time.perf_counter()

    print(f"\n--- Evaluating Baseline on {split_name.upper()} ({len(tickets)} tickets) ---")
    for idx, item in enumerate(tickets, start=1):
        t0 = time.perf_counter()
        try:
            pred = module(ticket_text=item["ticket_text"])
            pred_cat = pred.category
            pred_ord = pred.order_id
            pred_prio = pred.priority
            error_msg = None
        except Exception as exc:
            pred_cat = "Error"
            pred_ord = None
            pred_prio = "Error"
            error_msg = str(exc)

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
            "error": error_msg,
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
        "split": split_name,
        "metrics": metrics,
        "predictions": records,
    }


def main() -> None:
    check_ollama_server()

    lm = get_ollama_lm(temperature=0.0, max_tokens=300)
    dspy.configure(lm=lm)

    baseline_module = TicketClassifierModule(use_cot=True)

    val_results = evaluate_split(baseline_module, Path("data/validation.json"), "validation")
    test_results = evaluate_split(baseline_module, Path("data/test.json"), "test")

    output = {
        "model": "ollama_chat/qwen2.5:1.5b",
        "method": "Baseline (Uncompiled ChainOfThought)",
        "validation": val_results,
        "test": test_results,
    }

    out_path = Path("results/baseline_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n================ BASELINE SUMMARY ================")
    for split_key in ["validation", "test"]:
        m = output[split_key]["metrics"]
        print(f"\n[{split_key.upper()}] (Runtime: {m['runtime_sec']}s | Avg: {m['avg_latency_per_ticket_sec']}s/ticket)")
        print(f"  Exact Match Accuracy : {m['exact_match_accuracy']:.4f} ({m['exact_match_count']}/{m['total_samples']})")
        print(f"  Category Accuracy    : {m['category_metrics']['accuracy']:.4f} | Macro F1: {m['category_metrics']['macro_f1']:.4f}")
        print(f"  Priority Accuracy    : {m['priority_metrics']['accuracy']:.4f} | Macro F1: {m['priority_metrics']['macro_f1']:.4f}")
        print(f"  Order ID (Present)   : {m['order_id_metrics']['present_only_accuracy']:.4f} ({m['order_id_metrics']['present_only_exact_matches']}/{m['order_id_metrics']['present_only_count']}) | Entity F1: {m['order_id_metrics']['entity_f1']:.4f}")
    print(f"\nSaved full baseline results to: {out_path}")


if __name__ == "__main__":
    main()

