import json
import random
from collections import Counter, defaultdict
from pathlib import Path

RANDOM_SEED = 42

def summarize_split(records):
    return {
        "count": len(records),
        "categories": dict(sorted(Counter(r["category"] for r in records).items())),
        "priorities": dict(sorted(Counter(r["priority"] for r in records).items())),
        "with_order_id": sum(1 for r in records if r["order_id"] is not None),
        "without_order_id": sum(1 for r in records if r["order_id"] is None),
    }

def main():
    clean_path = Path("data/clean_tickets.json")
    with clean_path.open("r", encoding="utf-8") as f:
        tickets = json.load(f)

    rng = random.Random(RANDOM_SEED)

    # Group tickets by (category, priority) for exact stratified splitting
    strata = defaultdict(list)
    for t in tickets:
        strata[(t["category"], t["priority"])].append(t)

    train_set, val_set, test_set = [], [], []

    for key in sorted(strata.keys()):
        group = list(strata[key])
        rng.shuffle(group)
        # Each stratum has 4 tickets: 2 for Train, 1 for Validation, 1 for Test
        train_set.extend(group[:2])
        val_set.extend(group[2:3])
        test_set.extend(group[3:4])

    rng.shuffle(train_set)
    rng.shuffle(val_set)
    rng.shuffle(test_set)

    # Strict Data Leakage Check
    train_ids = {t["id"] for t in train_set}
    val_ids = {t["id"] for t in val_set}
    test_ids = {t["id"] for t in test_set}
    all_ids = {t["id"] for t in tickets}

    assert train_ids.isdisjoint(val_ids), "Data leakage detected between Train and Val!"
    assert train_ids.isdisjoint(test_ids), "Data leakage detected between Train and Test!"
    assert val_ids.isdisjoint(test_ids), "Data leakage detected between Val and Test!"
    assert (train_ids | val_ids | test_ids) == all_ids, "Mismatch in total split records!"

    # Save splits
    splits = {
        "data/train.json": train_set,
        "data/validation.json": val_set,
        "data/test.json": test_set,
    }
    for path_str, data in splits.items():
        with Path(path_str).open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    split_report = {
        "random_seed": RANDOM_SEED,
        "split_ratio": "50% Train / 25% Validation / 25% Test",
        "stratification_keys": ["category", "priority"],
        "data_leakage_verified": True,
        "statistical_note": (
            "Each split is strictly balanced across all 5 categories and 3 priorities. "
            "Because the test set contains 15 samples (each sample = 6.67%), metrics serve as "
            "controlled prototype benchmarks rather than large-scale statistical guarantees."
        ),
        "splits": {
            "train": summarize_split(train_set),
            "validation": summarize_split(val_set),
            "test": summarize_split(test_set),
        },
    }

    report_path = Path("data/split_report.json")
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(split_report, f, indent=2, ensure_ascii=False)

    print("=== STRATIFIED DATASET SPLIT COMPLETE ===")
    print(f"Random Seed        : {RANDOM_SEED}")
    print(f"Data Leakage Check : PASSED (0 overlapping IDs)")
    for name, info in split_report["splits"].items():
        print(f"\n[{name.upper()}] Total = {info['count']}")
        print(f"  Categories : {info['categories']}")
        print(f"  Priorities : {info['priorities']}")
        print(f"  Order IDs  : present={info['with_order_id']}, null={info['without_order_id']}")
    print(f"\nSaved split report to: {report_path}")

if __name__ == "__main__":
    main()

