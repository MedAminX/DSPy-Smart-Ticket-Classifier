
# Smart Customer Support — AI Ticket Classifier (Stanford DSPy)



**Academic Module:** AI Engineering and Software Architecture in the Era of AI and AI Agents (DS1 — Project 10)  

**Institution:** Institut Supérieur de Gestion de Tunis (ISG Tunis) — 2026–2027  



---



## 1. Project Overview



This project implements a declarative, self-optimizing customer support ticket classification prototype using **Stanford DSPy** paired with a local **Ollama (`qwen2.5:1.5b`)** language model, strict **Pydantic v2** schema contracts, and an interactive **Streamlit** web interface.



Given an unstructured customer message, the system extracts and validates three structured fields:

- **`category`**: `Shipping` | `Refund` | `Payment` | `Technical` | `Other`

- **`order_id`**: Explicit order identifier matching `^ORD-\d{4}$` or `null` (`None`)

- **`priority`**: `High` | `Medium` | `Low`



---



## 2. System Architecture & Output Sanitization



```text

[Streamlit Web UI (src/app.py)]

        │

        ▼  TicketRequest (Pydantic v2 Strict Input Validation)

[DSPy Integration Adapter (src/dspy_adapter.py)]

        │

        ▼  Compiled ChainOfThought Artifact (artifacts/optimized_ticket_classifier.json)

[Local LLM Runtime: Ollama — qwen2.5:1.5b]

        │

        ▼  Deterministic Sanitizer + TicketResult (Pydantic v2 Output Gate)

[Structured Output & Error Handling Display]

```



### Defensive Output Sanitization (`src/dspy_adapter.py`)

Before passing raw LLM predictions to the strict `Pydantic v2` output contract (`TicketResult`), `src/dspy_adapter.py` applies deterministic normalization:

- Any unrecognized `category` falls back to `"Other"`.

- Any unrecognized `priority` falls back to `"Medium"`.

- Any `order_id` string that does not match `^ORD-\d{4}$` is normalized to `None`.



*Engineering note:* This defensive layer guarantees runtime stability and prevents malformed LLM strings from crashing the UI, though it masks raw format deviations rather than exposing them as validation failures.



---



## 3. Repository Structure



```text

DSPy_Project/

├── artifacts/

│   └── optimized_ticket_classifier.json   # Compiled DSPy ChainOfThought artifact (ClassBalanced_FewShot_5)

├── data/

│   ├── raw_tickets.json                   # Initial synthetic ticket dataset (60 tickets)

│   ├── clean_tickets.json                 # Deduplicated, validated, class-balanced dataset (60 tickets)

│   ├── dataset_quality_report.json        # Automated data quality & balance audit report

│   ├── train.json                         # Stratified training split (30 tickets — 50%)

│   ├── validation.json                    # Stratified validation split (15 tickets — 25%)

│   ├── test.json                          # Held-out test split (15 tickets — 25%)

│   └── split_report.json                  # Split distribution & leakage check report

├── results/

│   ├── baseline_results.json              # Zero-shot baseline evaluation metrics

│   ├── optimization_report.json           # LabeledFewShot vs. BootstrapFewShot selection log

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

│   ├── optimize_dspy.py                   # Teleprompter comparison & artifact serializer

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

# 1. Navigate to the cloned/extracted repository root (e.g., DSPy_Project or DSPy-Smart-Ticket-Classifier)

cd <repository-folder>



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

python -m src.run_miprov2_ablation

python -m src.compare_before_after

```



---



## 6. Evaluation Results & Optimization Analysis



### A. Held-Out Test Set Comparison (`results/before_after_comparison.json` — 15 Tickets)



| Metric | Zero-Shot Baseline | Compiled DSPy (`ClassBalanced_FewShot_5`) | Delta |

| :--- | :---: | :---: | :---: |

| **Exact Match Accuracy** | `20.00%` (3/15) | **`46.67%`** (7/15) | **`+26.67%`** |

| **Category Accuracy** | `66.67%` (10/15) | **`86.67%`** (13/15) | **`+20.00%`** |

| **Category Macro-F1** | `0.6733` | **`0.8629`** | **`+0.1896`** |

| **Priority Accuracy** | `53.33%` (8/15) | `53.33%` (8/15) | `0.00%` |

| **Priority Macro-F1** | `0.4306` | `0.4222` | `-0.0084` |

| **Order ID Entity Precision** | `77.78%` | **`100.00%`** | **`+22.22%`** |

| **Order ID Entity Recall** | `100.00%` | `57.14%` | `-42.86%` |

| **Order ID Entity F1** | `0.8750` | `0.7273` | `-0.1477` |

| **Order ID Null Specificity** | `75.00%` | **`100.00%`** | **`+25.00%`** |

| **Avg Latency / Ticket** | `7.164s` | `8.159s` | `+0.995s` |



### B. Empirical Findings & Trade-Offs

1. **Substantial Gains in Primary Classification:** Compilation more than doubled end-to-end **Exact Match Accuracy** (`+26.67%`) and raised **Category Accuracy** to `86.67%` (`Macro-F1 = 0.8629`).

2. **Precision-vs-Recall Trade-off on `Order ID`:** Eliminated false-positive order IDs completely (`100.00%` Precision and `100.00%` Null Specificity), at the cost of lower recall (`57.14%`) on explicit IDs due to the compact `1.5B` parameter capacity.

3. **Priority Bottleneck:** Subjective urgency estimation (`High` / `Medium` / `Low`) remained unchanged at `53.33%`, indicating a natural boundary for a `1.5B` local model without domain-specific urgency rubrics.

4. **Prototype Scope:** The 15-ticket held-out test set provides a controlled educational benchmark rather than a large-scale production statistical guarantee.



### C. Teleprompter Comparison & `MIPROv2` Ablation (`Validation Set` — 15 Tickets)

`src/optimize_dspy.py` systematically compares **`LabeledFewShot` (`ClassBalanced_FewShot_5`)** and **`BootstrapFewShot` (`BootstrapFewShot_Stratified15`)** on the validation split and serializes the highest-scoring candidate (`ClassBalanced_FewShot_5`) to `artifacts/optimized_ticket_classifier.json`. In addition, `src/run_miprov2_ablation.py` benchmarks **`MIPROv2_Light`** (`results/miprov2_ablation_report.json`):



| Strategy / Teleprompter | Underlying Optimizer | Val Composite | Val Exact Match | Val Category Acc | Val Order F1 | Compile Time | Active Demos |

| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |

| `Baseline (Zero-Shot)` | None | — | `13.33%` | `33.33%` | `0.8421` | `0.000s` | `0` |

| **`ClassBalanced_FewShot_5` (Selected)** | `dspy.LabeledFewShot` | **`0.5064`** | **`33.33%`** | **`73.33%`** | `0.5455` | **`0.003s`** | `5` |

| `BootstrapFewShot_Stratified15` | `dspy.BootstrapFewShot` | `0.4868` | `26.67%` | `60.00%` | `0.8571` | `151.938s` | `5` |

| `MIPROv2_Light` (Ablation) | `dspy.MIPROv2` (3 Optuna trials) | — | `33.33%` | `66.67%` | `0.8889` | `820.357s` | `0` |



**Why `ClassBalanced_FewShot_5` was selected:** On a CPU-bound environment (`Intel Core i3`, `8 GB RAM`, `qwen2.5:1.5b`), `ClassBalanced_FewShot_5` achieved the highest validation composite score (`0.5064`) and category accuracy (`73.33%`) in `0.003s`. `BootstrapFewShot` and `MIPROv2` are preserved as documented ablation experiments showing that while bootstrapped reasoning traces improve `order_id` extraction, they degrade category accuracy and require heavy compilation times (`2.5 min` and `13.7 min`, respectively) on CPU hardware.

