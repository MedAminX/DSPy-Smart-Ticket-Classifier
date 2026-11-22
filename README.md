# DSPy Smart Ticket Classifier

A local customer-support ticket classifier built with **Stanford DSPy**, **Ollama**, **Qwen 2.5 1.5B**, **Pydantic**, and **Streamlit**.

**Course:** DS1 – Project 10: *Declarative Self-Optimizing Language Programs with Stanford DSPy (MIPROv2)*  
**Institution:** ISG Tunis · **Academic year:** 2026–2027

## 1. Project Objective

Classify a customer-support message and extract its order ID into a structured result:

| Field | Values |
|---|---|
| Category | `Shipping`, `Refund`, `Payment`, `Technical`, `Other` |
| Order ID | `ORD-XXXX` when present; otherwise `null` |
| Priority | `High`, `Medium`, `Low` |

The language model runs locally through Ollama. DSPy defines the prediction program and its optimization experiments; Pydantic validates the application's input and output.

## 2. Architecture

```text
Customer Ticket
      |
      v
Streamlit Interface
      |
      v
Pydantic Input Validation
      |
      v
DSPy Adapter
      |
      v
Compiled DSPy ChainOfThought Program
      |
      v
Ollama – Qwen 2.5 1.5B
      |
      v
Output Normalization
      |
      v
Pydantic Output Validation
      |
      v
Category + Order ID + Priority
```

## 3. DSPy Implementation

- **Signature:** `TicketClassificationSignature` defines the ticket input and the three prediction fields.
- **Module:** `TicketClassifierModule` uses `dspy.ChainOfThought` to produce the structured prediction.
- **Baseline:** zero-shot DSPy prediction.
- **Optimization experiments:** `dspy.LabeledFewShot`, `dspy.BootstrapFewShot`, and `dspy.MIPROv2`.
- **Primary artifact:** `ClassBalanced_FewShot_5`, selected using the Validation Set and saved as `artifacts/optimized_ticket_classifier.json`.

The selected artifact uses five balanced labeled demonstrations. `BootstrapFewShot` and `MIPROv2` were also executed and their results were retained as comparison experiments. The primary artifact is **not** a BootstrapFewShot or MIPROv2 artifact.

## 4. Dataset and Evaluation

The prepared dataset contains **60 synthetic tickets**, balanced across five categories and three priority levels.

| Split | Samples | Purpose |
|---|---:|---|
| Train | 30 | Labeled examples for program/optimizer development |
| Validation | 15 | Compare candidates and select the primary configuration |
| Test | 15 | Held-out Before/After evaluation |

The Validation Set is used for candidate selection. The Test Set is reserved for the final comparison.

### Evaluation metrics

- Exact Match Accuracy
- Category Accuracy and Macro-F1
- Priority Accuracy and Macro-F1
- Order ID entity Precision, Recall, and F1
- Order ID null specificity
- Runtime and average latency

## 5. Optimization Ablation — Validation Set

Recorded results from the project's optimization reports:

| Candidate | Method | Exact Match | Category Accuracy | Compile time |
|---|---|---:|---:|---:|
| Baseline | Zero-shot | 13.33% | 33.33% | 0.000 s |
| **ClassBalanced_FewShot_5 — Selected** | `LabeledFewShot` | **33.33%** | **73.33%** | **0.003 s** |
| BootstrapFewShot_Stratified15 | `BootstrapFewShot` | 26.67% | 60.00% | 151.938 s |
| MIPROv2_Light | `MIPROv2` | 33.33% | 66.67% | 820.357 s |

The selection is based on the Validation Set composite score recorded by `src/optimize_dspy.py`:

| Candidate | Validation composite |
|---|---:|
| ClassBalanced_FewShot_5 | **0.5064** |
| BootstrapFewShot_Stratified15 | 0.4868 |

`ClassBalanced_FewShot_5` was retained because it achieved the higher recorded composite score and category accuracy in this validation comparison. MIPROv2 is documented separately as an ablation experiment; it did not replace the primary artifact.

## 6. Before/After Results — Held-Out Test Set

The following results are from the recorded evaluation of **15 test tickets**.

| Metric | Zero-shot baseline | Selected artifact |
|---|---:|---:|
| Exact Match Accuracy | 20.00% | **46.67%** |
| Category Accuracy | 66.67% | **86.67%** |
| Category Macro-F1 | 0.6733 | **0.8629** |
| Priority Accuracy | 53.33% | 53.33% |
| Priority Macro-F1 | 0.4306 | 0.4222 |
| Order ID Precision | 77.78% | **100.00%** |
| Order ID Recall | 100.00% | 57.14% |
| Order ID F1 | 0.8750 | 0.7273 |
| Order ID Null Specificity | 75.00% | **100.00%** |
| Average latency / ticket | 7.164 s | 8.159 s |

**Zero hallucination on missing Order IDs (in this test set):** the selected artifact achieved **100.00% Order ID Null Specificity (8/8)**, correctly returning `null` for all eight test tickets that did not contain an order ID. This result is limited to the evaluated test examples and does not imply that the model can never hallucinate an order ID.

Category classification and exact match improved in this run. Priority accuracy did not improve, and Order ID recall decreased. Therefore, the results do **not** show improvement on every metric.

## 7. Project Structure

```text
DSPy-Smart-Ticket-Classifier/
├── .gitignore
├── README.md
├── requirements.txt
│
├── artifacts/                         # Serialized DSPy classifier artifact
│   └── optimized_ticket_classifier.json
│
├── data/                              # Dataset files and preparation reports
│   ├── raw_tickets.json
│   ├── clean_tickets.json
│   ├── dataset_quality_report.json
│   ├── split_report.json
│   ├── train.json
│   ├── validation.json
│   └── test.json
│
├── results/                           # Recorded evaluation and optimization results
│   ├── baseline_results.json
│   ├── before_after_comparison.json
│   ├── optimization_report.json
│   └── miprov2_ablation_report.json
│
└── src/                               # Application, DSPy pipeline, data tools, and tests
    ├── app.py
    ├── schemas.py
    ├── signatures.py
    ├── modules.py
    ├── dspy_adapter.py
    ├── metrics.py
    ├── mock_classifier.py
    ├── create_raw_dataset.py
    ├── validate_and_prepare_dataset.py
    ├── split_dataset.py
    ├── run_baseline.py
    ├── optimize_dspy.py
    ├── run_miprov2_ablation.py
    ├── compare_before_after.py
    └── test_schemas.py
```

### Directory Overview

| Path | Purpose |
|---|---|
| `artifacts/` | Saved, optimized DSPy program used by the application. |
| `data/` | Raw and prepared ticket datasets, train/validation/test splits, and dataset reports. |
| `results/` | Baseline, optimizer ablation, and held-out Before/After evaluation reports. |
| `src/` | Streamlit application, Pydantic schemas, DSPy signatures/modules, adapter, metrics, dataset preparation scripts, optimization scripts, and tests. |
| `requirements.txt` | Python dependencies required by the project. |
| `.gitignore` | Files and directories excluded from Git tracking. |

## 8. Setup and Run

### Requirements

- Python 3.11
- Ollama
- `qwen2.5:1.5b`
- Dependencies listed in `requirements.txt`

Download the model once:

```powershell
ollama pull qwen2.5:1.5b
```

Create the environment and install dependencies from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Start the application:

```powershell
streamlit run src/app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## 9. Reproduce Evaluation and Tests

Run the recorded evaluation script:

```powershell
python -m src.compare_before_after
```

Run the unit and interface tests:

```powershell
python -m unittest discover -s src -p "test_*.py" -v
```

The captured test run completed **21 tests in 5.043 seconds** with `OK`. Streamlit emitted `missing ScriptRunContext` warnings during bare-mode UI tests; the captured run still passed.

Optimizer compilation can take substantially longer than normal inference, especially MIPROv2. The JSON reports in `results/` preserve the recorded experiment results; rerunning experiments may produce different timings and metrics.

## 10. Scope and Limitations

- The dataset is synthetic and small (60 tickets; 15 test examples).
- The local model may produce classification or extraction errors.
- The selected configuration improves some test metrics but reduces Order ID recall and does not improve every metric.
- This is an academic prototype, not a production-ready support automation system.

## References

- [Stanford DSPy Documentation](https://dspy.ai/)
- [Stanford DSPy GitHub](https://github.com/stanfordnlp/dspy)
- [Ollama](https://ollama.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Streamlit Documentation](https://docs.streamlit.io/)
