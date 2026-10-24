# How the code works (plain version)

The whole project is one idea: show a small model hundreds of worked examples until it can do one narrow job by itself.

```
PDF orders  ->  text  ->  Gemini writes the "answer key"  ->  split train / test
                                                                   |
                         base model scored on test  <--------------+
                         fine-tune on train (QLoRA)
                         fine-tuned model scored on test  -->  results table
```

## The files, in the order the data flows

### `src/fetch_open_orders.py` (Step 0: get real orders)
Downloads about 400 public Telangana High Court orders from an open dataset on AWS. It reads a metadata table first (one row per judgment), picks one order per case so the train/test split later has many different cases, skips big files (long judgments, heavy scans), and saves each PDF as `<CNR>_order_<n>_<date>.pdf`, the same naming Case Intel uses. It also writes `manifest.csv` (where each file came from) and `SOURCE.md` (the CC-BY-4.0 credit).

### `src/extract_text.py` (Step 1: PDF to text)
Pulls the text out of each PDF with `pypdf`. If a PDF is a scan with no text layer, it runs OCR (Tesseract). It removes page numbers and extra spaces, and drops duplicates by hashing the cleaned text (SHA-256). Output: `orders.jsonl`, one order per line.

### `src/schema.py` (what a correct answer looks like)
The single definition of the JSON we want: `case_number`, `order_date`, `next_hearing_date`, `outcome` (one of 9 fixed values), `directions` (a list of who must do what by when) and `summary`.
- `validate()` checks any JSON against that definition and lists the problems.
- `parse_model_output()` pulls JSON out of whatever a model wrote, and tells you whether the raw output was clean JSON or needed rescuing.
- `canonical()` writes the answer in one fixed format, used as the training target.

### `src/prompt.py` (the question we ask)
One system prompt and one `messages()` function, used by labelling, training and evaluation. That matters: the model is always asked the same question in the same words, so what it learns in training is exactly what it is tested on. `clip()` keeps the first 45% and the last part of long orders, because the case number is at the top and the operative directions and next date are at the bottom.

### `src/label_gemini.py` (Step 2: the teacher writes the answer key)
Sends each order to Gemini with JSON-schema mode on and temperature 0, so the reply is always parseable and repeatable. Every label goes through `validate()`. Records are saved as `ok`, `rejected` or `error`. It is resumable (re-running skips finished orders, retries errors) and stops cleanly when the free quota runs out.

### `src/build_dataset.py` (Step 3: train/test split)
Joins orders with labels and splits **by case, not by order**. A hash of the CNR decides the side, so every order of one case lands on the same side. Otherwise two orders of the same case (very similar text) could end up one in train and one in test, and the test would be too easy. It writes chat-format rows (`system`, `user`, `assistant`) and prints the case overlap, which must be 0. If you hand-correct a label in `data/review.jsonl`, that overrides Gemini.

### `src/metrics.py` (Step 6: the exam marking)
Scores predictions against the held-out labels: valid JSON, schema-valid, outcome, order date, next hearing date, case number, directions F1, seconds per order. Anything the model fails to parse counts as wrong on every field (it is never skipped, so a model can't look better by failing quietly). A direction counts as correct if the party matches and at least half the words of the action overlap.

### `notebooks/court_order_qlora.ipynb` (the runner, built by `make_notebook.py`)
Runs the steps above in Colab, plus the three model steps:
- **Baseline:** load Llama 3.2 1B (4-bit), ask it the same question with no training, save its answers.
- **Fine-tune (QLoRA):**
  - the base model is frozen and shrunk to 4 bits to fit a free T4;
  - small trainable "LoRA" layers (about 1% of the size, rank 16) are attached to every linear layer;
  - the loss is computed only on the JSON answer, not on the order text (`train_on_responses_only`), so the model learns to write answers, not to memorise orders;
  - learning rate 2e-4, effective batch size 8, 3 epochs, 120 steps in total.
- **Evaluate and export:** the same 78 held-out orders, then a 4-bit GGUF file plus an Ollama `Modelfile` so it can run locally.

### `tests/`
20 tests that run without a GPU: schema validation, the metric maths, the split (no case overlap), the labeller's retry logic, and the fetch script's filters.

## Why these choices (good interview answers)

| Choice | Reason |
|---|---|
| 1B model, not bigger | Cheap to run, fits a free GPU, and a narrow task doesn't need more |
| QLoRA, not full fine-tune | Trains about 1% of the weights; fits in free-tier memory; the adapter is small |
| Gemini as teacher | Labelling 400 orders by hand is slow; a validated teacher is fast and consistent |
| Split by case | Prevents leakage between train and test |
| Loss on the answer only | The model learns the output format, not the input text |
| Greedy decoding, temperature 0 | Reproducible numbers |
| Custom metrics | ROUGE says nothing about whether the hearing date is right |

## Honest limits
- The answer key is Gemini's, so scores mean "agrees with Gemini", not "is right".
- Next-hearing-date accuracy is flattered because most orders have no next hearing, so `null` is usually correct.
- 78 test orders is a small exam: one order is 1.3 points.
- Directions F1 (66.7) is the weak field.
- Not yet connected to Case Intel (see `CASE_INTEL_INTEGRATION.md`).
