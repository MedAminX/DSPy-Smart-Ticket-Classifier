# Smart Customer Support — AI Ticket Classifier (Stanford DSPy)

**Academic Module:** AI Engineering and Software Architecture in the Era of AI and AI Agents (DS1 — Project 10)  
**Institution:** Institut Supérieur de Gestion de Tunis (ISG Tunis) — 2026–2027  

---

## 1. Project Overview

This project implements a declarative, self-optimizing customer support ticket classification system using **Stanford DSPy** paired with a local **Ollama (`qwen2.5:1.5b`)** language model, strict **Pydantic v2** schema contracts, and an interactive **Streamlit** web interface.

Given an unstructured customer message, the system extracts and validates three structured fields:
- **`category`**: `Shipping` | `Refund` | `Payment` | `Technical` | `Other`
- **`order_id`**: Explicit order identifier matching `^ORD-\d{4}$` or `null` (`None`)
- **`priority`**: `High` | `Medium` | `Low`

---

## 2. System Architecture

```text
[Streamlit Web UI (src/app.py)]
        │
        ▼  TicketRequest (Pydantic v2 Strict Input Validation)
[DSPy Integration Adapter (src/dspy_adapter.py)]
        │
        ▼  Compiled ChainOfThought Artifact (artifacts/optimized_ticket_classifier.json)
[Local LLM Runtime: Ollama — qwen2.5:1.5b]
        │
        ▼  Deterministic Regex/Literal Sanitizer + TicketResult (Pydantic v2 Gate)
[Structured Output & Error Handling Display]
```

---

## 3. Repository Structure

```text
DSPy_Project/
├── artifacts/
│   └── optimized_ticket_classifier.json   # Compiled DSPy ChainOfThought program
├── data/
│   ├── raw_tickets.json                   # Initial synthetic ticket dataset (75 tickets)
│   ├── clean_tickets.json                 # Deduplicated, validated, class-balanced dataset
│   ├── dataset_quality_report.json        # Automated data quality & balance audit report
│   ├── train.json                         # Stratified training split (45 tickets)
│   ├── validation.json                    # Stratified validation split (15 tickets)
│   ├── test.json                          # Held-out test split (15 tickets)
│   └── split_report.json                  # Split distribution & leakage check report
├── results/
│   ├── baseline_results.json              # Zero-shot baseline evaluation metrics
│   ├── optimization_report.json           # Candidate optimizer selection report
│   ├── miprov2_ablation_report.json       # MIPROv2 + Optuna ablation benchmark
│   └── before_after_comparison.json       # Held-out Test Set comparison (Before vs. After)
├── src/
│   ├── app.py                             # Streamlit interactive web application (Live DSPy Mode)
│   ├── schemas.py                         # Strict Pydantic v2 input/output contracts
│   ├── dspy_adapter.py                    # Production inference wrapper & deterministic sanitizer
│   ├── signatures.py                      # Declarative dspy.Signature definition
│   ├── modules.py                         # Composable dspy.ChainOfThought module & Ollama LM setup
│   ├── metrics.py                         # Exact-match, Macro-F1 & Order-ID entity evaluators
│   ├── create_raw_dataset.py              # Reproducible raw dataset generator
│   ├── validate_and_prepare_dataset.py    # Data cleaning & quality verification pipeline
│   ├── split_dataset.py                   # Stratified Train/Validation/Test splitter
│   ├── run_baseline.py                    # Zero-shot baseline runner
│   ├── optimize_dspy.py                   # Teleprompter compilation & artifact serializer
│   ├── run_miprov2_ablation.py            # MIPROv2 Bayesian optimization ablation study
│   ├── compare_before_after.py            # Deterministic Before vs. After test evaluator
│   ├── mock_classifier.py                 # Initial mock stub used during UI prototyping
│   └── test_schemas.py                    # Automated unit & Streamlit AppTest suite (21 tests)
├── README.md                              # Unified project documentation
└── requirements.txt                       # Pinned Python dependencies
```

---

## 4. Installation & Environment Setup (Windows PowerShell)

### Prerequisites
1. **Python 3.11+**
2. **Ollama** installed locally with the `qwen2.5:1.5b` model pulled:
   ```powershell
   ollama pull qwen2.5:1.5b
   ```

### Setup Steps
```powershell
# 1. Navigate to the project root
cd DSPy_Project

# 2. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 5. Running the Application & Verification Suite

### A. Run the Automated Test Suite (21 Unit & UI Contract Tests)
```powershell
python -m unittest discover -s src -p "test_*.py" -v
```

### B. Launch the Interactive Streamlit Web App
Ensure Ollama is running in the background, then start the UI:
```powershell
streamlit run src\app.py
```
Open `http://localhost:8501` in your browser to classify customer tickets in real time.

### C. Reproduce the DSPy Compilation & Benchmark Pipeline (Optional)
```powershell
python -m src.validate_and_prepare_dataset
python -m src.split_dataset
python -m src.run_baseline
python -m src.optimize_dspy
python -m src.compare_before_after
```

---

## 6. Evaluation Results (Held-Out Test Set — 15 Tickets)

| Metric | Zero-Shot Baseline | Compiled DSPy (`ClassBalanced_FewShot_5`) | Improvement |
| :--- | :---: | :---: | :---: |
| **Exact Match Accuracy** | `20.00%` | **`46.67%`** | **`+26.67%`** |
| **Category Accuracy** | `40.00%` | **`86.67%`** | **`+46.67%`** |
| **Category Macro-F1** | `0.3695` | **`0.8622`** | **`+0.4927`** |
| **Priority Accuracy** | `53.33%` | **`66.67%`** | **`+13.34%`** |
| **Order ID Precision** | `57.14%` | **`100.00%`** | **`+42.86%`** |
| **Order ID Entity F1** | `0.5714` | **`0.7273`** | **`+0.1559`** |

### Key Ablation Takeaway (`MIPROv2` vs. `ClassBalanced_FewShot_5`)
On a CPU-bound local environment (`Intel Core i3`, `8 GB RAM`, `qwen2.5:1.5b`), `MIPROv2` with `Optuna` required `820.36s` (~13.7 min) for 3 Bayesian trials and reverted to 0 demonstrations because the compact `1.5B` model could not synthesize stronger meta-instructions. In contrast, `ClassBalanced_FewShot_5` compiled in `0.003s` and achieved superior category classification (`86.67%` Test) while eliminating `ORD-XXXX` hallucinations (`100%` Precision).
