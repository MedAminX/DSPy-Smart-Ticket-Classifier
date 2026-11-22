import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import dspy
from dspy.teleprompt import MIPROv2
from src.metrics import dspy_ticket_metric, evaluate_predictions
from src.modules import TicketClassifierModule, get_ollama_lm

CATEGORIES_ORDER = ["Shipping", "Refund", "Payment", "Technical", "Other"]


def load_stratified_examples(json_path: Path, limit_per_class: int | None = None) -> List[dspy.Example]:
    with json_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    selected: List[Dict[str, Any]] = []
    max_len = max(len(by_cat[c]) for c in CATEGORIES_ORDER)
    cap = limit_per_class if limit_per_class is not None else max_len

    for i in range(min(max_len, cap)):
        for c in CATEGORIES_ORDER:
            if i < len(by_cat[c]):
                selected.append(by_cat[c][i])

    return [
        dspy.Example(
            ticket_text=r["ticket_text"],
            category=r["category"],
            order_id=r["order_id"],
            priority=r["priority"],
        ).with_inputs("ticket_text")
        for r in selected
    ]


def evaluate_on_validation(module: dspy.Module, val_path: Path) -> Dict[str, Any]:
    with val_path.open("r", encoding="utf-8") as f:
        tickets = json.load(f)

    records = []
    t_start = time.perf_counter()
    print(f"\n--- Evaluating MIPROv2_Light on VALIDATION ({len(tickets)} tickets) ---")
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

    runtime = time.perf_counter() - t_start
    metrics = evaluate_predictions(records)
    metrics["runtime_sec"] = round(runtime, 3)
    metrics["avg_latency_per_ticket_sec"] = round(runtime / len(tickets), 3) if tickets else 0.0
    return metrics


def main() -> None:
    lm = get_ollama_lm(temperature=0.0, max_tokens=250)
    dspy.configure(lm=lm)

    # Use 15 stratified train samples (3 per category) + 15 validation samples
    trainset_15 = load_stratified_examples(Path("data/train.json"), limit_per_class=3)
    valset_15 = load_stratified_examples(Path("data/validation.json"))
    val_path = Path("data/validation.json")

    print("================ RUNNING MIPROv2 LIGHT ABLATION ================")
    print(f"Train samples: {len(trainset_15)} (stratified) | Validation samples: {len(valset_15)}")

    t0 = time.perf_counter()
    student = TicketClassifierModule(use_cot=True)
    mipro_optimizer = MIPROv2(
        metric=dspy_ticket_metric,
        auto=None,
        num_candidates=2,
        init_temperature=0.5,
        max_bootstrapped_demos=3,
        max_labeled_demos=5,
        num_threads=1,
    )
    compiled_mipro = mipro_optimizer.compile(
        student=student,
        trainset=trainset_15,
        valset=valset_15,
        num_trials=3,
        minibatch=False,
    )
    compile_runtime_sec = round(time.perf_counter() - t0, 3)

    inner_predict = getattr(compiled_mipro.predictor, "predict", compiled_mipro.predictor)
    active_demos = len(getattr(inner_predict, "demos", []))
    final_instruction = getattr(getattr(inner_predict, "signature", None), "instructions", "")

    print(f"\nMIPROv2 compilation finished in {compile_runtime_sec}s | Active demos: {active_demos}")
    print(f"Compiled Instruction: {final_instruction}")

    mipro_val_metrics = evaluate_on_validation(compiled_mipro, val_path)

    # Load prior Validation benchmarks for direct comparison
    with Path("results/baseline_results.json").open("r", encoding="utf-8") as f:
        base_val = json.load(f)["validation"]["metrics"]
    with Path("results/optimization_report.json").open("r", encoding="utf-8") as f:
        bfs_val = json.load(f)["selected_validation_metrics"]

    ablation_summary = {
        "Baseline (Zero-Shot)": {
            "exact_match_accuracy": base_val["exact_match_accuracy"],
            "category_accuracy": base_val["category_metrics"]["accuracy"],
            "category_macro_f1": base_val["category_metrics"]["macro_f1"],
            "priority_accuracy": base_val["priority_metrics"]["accuracy"],
            "order_id_entity_f1": base_val["order_id_metrics"]["entity_f1"],
            "compile_time_sec": 0.0,
        },
        "ClassBalanced_FewShot_5 (Selected)": {
            "exact_match_accuracy": bfs_val["exact_match_accuracy"],
            "category_accuracy": bfs_val["category_metrics"]["accuracy"],
            "category_macro_f1": bfs_val["category_metrics"]["macro_f1"],
            "priority_accuracy": bfs_val["priority_metrics"]["accuracy"],
            "order_id_entity_f1": bfs_val["order_id_metrics"]["entity_f1"],
            "compile_time_sec": 0.003,
        },
        "MIPROv2_Light (Ablation)": {
            "exact_match_accuracy": mipro_val_metrics["exact_match_accuracy"],
            "category_accuracy": mipro_val_metrics["category_metrics"]["accuracy"],
            "category_macro_f1": mipro_val_metrics["category_metrics"]["macro_f1"],
            "priority_accuracy": mipro_val_metrics["priority_metrics"]["accuracy"],
            "order_id_entity_f1": mipro_val_metrics["order_id_metrics"]["entity_f1"],
            "compile_time_sec": compile_runtime_sec,
        },
    }

    report = {
        "experiment_type": "Optimizer Ablation Study on Validation Set",
        "model": "ollama_chat/qwen2.5:1.5b",
        "optuna_enabled": True,
        "hyperparameters": {
            "optimizer": "MIPROv2",
            "num_candidates": 2,
            "num_trials": 3,
            "max_bootstrapped_demos": 3,
            "max_labeled_demos": 5,
            "num_threads": 1,
            "minibatch": False,
        },
        "compile_runtime_sec": compile_runtime_sec,
        "active_demos_count": active_demos,
        "compiled_instruction": final_instruction,
        "validation_metrics": mipro_val_metrics,
        "validation_ablation_table": ablation_summary,
        "test_set_untouched": True,
        "primary_artifact_preserved": "artifacts/optimized_ticket_classifier.json",
    }

    out_path = Path("results/miprov2_ablation_report.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n================ OPTIMIZER ABLATION (VALIDATION SET) ================")
    print(f"{'Method':<35} | {'ExactMatch':>10} | {'CatAcc':>8} | {'OrderF1':>8} | {'Compile(s)':>10}")
    print("-" * 82)
    for name, m in ablation_summary.items():
        print(
            f"{name:<35} | {m['exact_match_accuracy']:>10.4f} | "
            f"{m['category_accuracy']:>8.4f} | {m['order_id_entity_f1']:>8.4f} | {m['compile_time_sec']:>10.3f}"
        )
    print(f"\nSaved ablation report to: {out_path}")


if __name__ == "__main__":
    main()
