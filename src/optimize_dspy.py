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
from dspy.teleprompt import BootstrapFewShot, LabeledFewShot
from src.metrics import dspy_ticket_metric, evaluate_predictions
from src.modules import TicketClassifierModule, get_ollama_lm

CATEGORIES_ORDER = ["Shipping", "Refund", "Payment", "Technical", "Other"]


def load_stratified_train_examples(json_path: Path) -> tuple[List[dspy.Example], List[dspy.Example]]:
    """Return full interleaved trainset and a strict 5-example class-balanced subset (1 per category)."""
    with json_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    interleaved: List[Dict[str, Any]] = []
    max_len = max(len(by_cat[c]) for c in CATEGORIES_ORDER)
    for i in range(max_len):
        for c in CATEGORIES_ORDER:
            if i < len(by_cat[c]):
                interleaved.append(by_cat[c][i])

    # Pick 1 clear example per category (mix of with/without order_id and priorities)
    one_per_cat = [by_cat[c][0] for c in CATEGORIES_ORDER]

    to_example = lambda r: dspy.Example(
        ticket_text=r["ticket_text"],
        category=r["category"],
        order_id=r["order_id"],
        priority=r["priority"],
    ).with_inputs("ticket_text")

    return [to_example(r) for r in interleaved], [to_example(r) for r in one_per_cat]


def count_active_demos(module: dspy.Module) -> int:
    inner = getattr(module.predictor, "predict", module.predictor)
    return len(getattr(inner, "demos", []))


def evaluate_on_validation(module: dspy.Module, val_path: Path, label: str) -> Dict[str, Any]:
    with val_path.open("r", encoding="utf-8") as f:
        tickets = json.load(f)

    records = []
    t_start = time.perf_counter()
    print(f"\n--- Validating {label} on VALIDATION ({len(tickets)} tickets) ---")
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


def score_validation(val_metrics: Dict[str, Any]) -> float:
    return round(
        0.40 * val_metrics["exact_match_accuracy"]
        + 0.25 * val_metrics["category_metrics"]["macro_f1"]
        + 0.20 * val_metrics["priority_metrics"]["macro_f1"]
        + 0.15 * val_metrics["order_id_metrics"]["entity_f1"],
        4,
    )


def main() -> None:
    lm = get_ollama_lm(temperature=0.0, max_tokens=250)
    dspy.configure(lm=lm)

    full_trainset, balanced_5_trainset = load_stratified_train_examples(Path("data/train.json"))
    val_path = Path("data/validation.json")

    experiment_logs: List[Dict[str, Any]] = []
    best_score = -1.0
    best_program = None
    best_summary: Dict[str, Any] = {}

    # Candidate 1: Class-Balanced LabeledFewShot (1 demo per category = 5 balanced demos)
    print("\n================ COMPILING: ClassBalanced_FewShot_5 =================")
    t0 = time.perf_counter()
    student_1 = TicketClassifierModule(use_cot=True)
    opt_1 = LabeledFewShot(k=5)
    prog_1 = opt_1.compile(student=student_1, trainset=balanced_5_trainset)
    t_compile_1 = round(time.perf_counter() - t0, 3)
    demos_1 = count_active_demos(prog_1)
    print(f"Compilation finished in {t_compile_1}s | Active balanced demos: {demos_1}")

    val_1 = evaluate_on_validation(prog_1, val_path, "ClassBalanced_FewShot_5")
    score_1 = score_validation(val_1)
    entry_1 = {
        "candidate_name": "ClassBalanced_FewShot_5",
        "hyperparameters": {
            "optimizer": "LabeledFewShot",
            "k": 5,
            "strict_class_balance": "1 demo per category (Shipping, Refund, Payment, Technical, Other)",
        },
        "compile_runtime_sec": t_compile_1,
        "active_demos_count": demos_1,
        "validation_composite_score": score_1,
        "validation_metrics": val_1,
    }
    experiment_logs.append(entry_1)
    print(
        f"\n[ClassBalanced_FewShot_5 VALIDATION] Composite={score_1:.4f} | "
        f"ExactMatch={val_1['exact_match_accuracy']:.4f} | "
        f"CatAcc={val_1['category_metrics']['accuracy']:.4f} (MacroF1={val_1['category_metrics']['macro_f1']:.4f}) | "
        f"PrioAcc={val_1['priority_metrics']['accuracy']:.4f} | "
        f"OrderF1={val_1['order_id_metrics']['entity_f1']:.4f}"
    )
    if score_1 > best_score:
        best_score, best_program, best_summary = score_1, prog_1, entry_1

    # Candidate 2: BootstrapFewShot initialized with balanced training subset (first 15 interleaved)
    print("\n================ COMPILING: BootstrapFewShot_Stratified15 =================")
    t0 = time.perf_counter()
    student_2 = TicketClassifierModule(use_cot=True)
    opt_2 = BootstrapFewShot(
        metric=dspy_ticket_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=5,
        max_rounds=1,
    )
    prog_2 = opt_2.compile(student=student_2, trainset=full_trainset[:15])
    t_compile_2 = round(time.perf_counter() - t0, 3)
    demos_2 = count_active_demos(prog_2)
    print(f"Compilation finished in {t_compile_2}s | Active demos: {demos_2}")

    val_2 = evaluate_on_validation(prog_2, val_path, "BootstrapFewShot_Stratified15")
    score_2 = score_validation(val_2)
    entry_2 = {
        "candidate_name": "BootstrapFewShot_Stratified15",
        "hyperparameters": {
            "optimizer": "BootstrapFewShot",
            "max_bootstrapped_demos": 3,
            "max_labeled_demos": 5,
            "max_rounds": 1,
            "trainset_subset": "15 stratified examples (3 per category)",
        },
        "compile_runtime_sec": t_compile_2,
        "active_demos_count": demos_2,
        "validation_composite_score": score_2,
        "validation_metrics": val_2,
    }
    experiment_logs.append(entry_2)
    print(
        f"\n[BootstrapFewShot_Stratified15 VALIDATION] Composite={score_2:.4f} | "
        f"ExactMatch={val_2['exact_match_accuracy']:.4f} | "
        f"CatAcc={val_2['category_metrics']['accuracy']:.4f} (MacroF1={val_2['category_metrics']['macro_f1']:.4f}) | "
        f"PrioAcc={val_2['priority_metrics']['accuracy']:.4f} | "
        f"OrderF1={val_2['order_id_metrics']['entity_f1']:.4f}"
    )
    if score_2 > best_score:
        best_score, best_program, best_summary = score_2, prog_2, entry_2

    artifact_path = Path("artifacts/optimized_ticket_classifier.json")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    best_program.save(str(artifact_path))

    report = {
        "model": "ollama_chat/qwen2.5:1.5b",
        "optimizers_evaluated": ["LabeledFewShot (5 class-balanced demos)", "BootstrapFewShot (15 stratified train samples)"],
        "optimization_rationale": (
            "Added explicit semantic definitions in TicketClassificationSignature and evaluated "
            "class-balanced few-shot selection (1 demo per category) against BootstrapFewShot on "
            "Validation Set to eliminate majority-class bias toward 'Technical'."
        ),
        "train_samples_used": len(full_trainset),
        "validation_samples_used": 15,
        "test_set_untouched": True,
        "selected_best_candidate": best_summary["candidate_name"],
        "selected_hyperparameters": best_summary["hyperparameters"],
        "selected_validation_metrics": best_summary["validation_metrics"],
        "all_validation_experiments": experiment_logs,
        "saved_artifact": str(artifact_path),
    }

    report_path = Path("results/optimization_report.json")
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n================ OPTIMIZATION COMPLETE ================")
    print(f"Selected Best Candidate : {best_summary['candidate_name']} (Score={best_score:.4f})")
    print(f"Saved Compiled Program  : {artifact_path}")
    print(f"Saved Optimization Log  : {report_path}")


if __name__ == "__main__":
    main()

