"""Generates notebooks/court_order_qlora.ipynb (kept as code so it is reviewable/diffable)."""
import json
import os

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# Court-Order Extractor: QLoRA fine-tune of Llama 3.2 1B

Turns an Indian court order into JSON: case number, order date, next hearing date, outcome, party-wise directions, summary.

**Runtime:** Runtime → Change runtime type → **T4 GPU** (free tier is enough). Total time with ~300 orders: about 1 to 1.5 hours, most of it Gemini labelling.

**Before you start**
1. Put this project folder in Google Drive at `MyDrive/Fine-Tuning-CaseIntel`. Step 0 downloads ~400 real Telangana High Court orders for you; any of your own order PDFs in `data/raw_orders/` are used too.
2. Add your Gemini key as a Colab secret: 🔑 icon in the left bar → name `GEMINI_API_KEY` → paste key → enable notebook access.
3. Run the cells top to bottom. Everything is saved to Drive, so if the runtime disconnects, re-run from the top: finished steps are skipped.
""")
code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
from google.colab import drive
drive.mount('/content/drive')
import os, sys
PROJECT = '/content/drive/MyDrive/Fine-Tuning-CaseIntel'
os.chdir(PROJECT); sys.path.insert(0, PROJECT)
os.makedirs('runs', exist_ok=True)
print(os.getcwd(), '| PDFs/TXTs in data/raw_orders:', sum(f.lower().endswith(('.pdf', '.txt')) for _, _, fs in os.walk('data/raw_orders') for f in fs))
""")
code("""
%%capture
!pip install unsloth pypdf google-genai pytesseract pdf2image
!apt-get -qq install -y tesseract-ocr poppler-utils
""")
md("""
## 0 · Get real court orders (open eCourts data)
Downloads ~400 public orders of the High Court for the State of Telangana from the open
[Indian High Court Judgments](https://registry.opendata.aws/indian-high-court-judgments/) dataset (CC-BY-4.0):
one order per case, short files only, renamed to Case Intel's `<CNR>_order_<n>_<date>.pdf` format.
Takes a few minutes and runs once (re-running only tops up). Writes `manifest.csv` and `SOURCE.md` next to the PDFs.
""")
code("""
!python -m src.fetch_open_orders data/raw_orders --n 400
""")
md("## 1 · PDFs → text\nScanned orders (no text layer) are OCR'd. Duplicates are dropped by SHA-256 of the normalised text.")
code("""
!python -m src.extract_text data/raw_orders data/orders.jsonl --ocr
""")
md("## 2 · Labels from Gemini (the teacher)\nJSON-schema mode, temperature 0, every label validated. Resumable: re-running only labels what is missing.")
code("""
from google.colab import userdata
os.environ['GEMINI_API_KEY'] = userdata.get('GEMINI_API_KEY')
!python -m src.label_gemini data/orders.jsonl data/labels.jsonl --model gemini-3.1-flash-lite
""")
md("""
### 2b · Spot-check the teacher (do this, it is the honest part)
Read ~20 labels against the order text. For any wrong one, add a line to `data/review.jsonl`:
`{"id": "...", "label": {...corrected...}}` or `{"id": "...", "drop": true}`. Corrections override Gemini in step 3.
Write down how many of the 20 were fully correct: that number goes in the README and on the resume line.
""")
code("""
import json, random
orders = {o['id']: o for o in map(json.loads, open('data/orders.jsonl'))}
latest = {r['id']: r for r in map(json.loads, open('data/labels.jsonl'))}
labels = [r for r in latest.values() if r['status'] == 'ok']
from collections import Counter
print(Counter(r['status'] for r in latest.values()), '| orders not labelled yet:', len(orders) - len(latest))
random.seed(7)
for r in random.sample(labels, min(20, len(labels))):
    t = orders[r['id']]['text']
    print('=' * 100, '\\n', r['id'], '\\n--- ORDER (start / end) ---\\n', t[:700], '\\n[...]\\n', t[-900:])
    print('--- LABEL ---\\n', json.dumps(r['label'], indent=1, ensure_ascii=False))
""")
md("## 3 · Train / test split (by case, so no test case is seen in training)")
code("""
!python -m src.build_dataset data/orders.jsonl data/labels.jsonl data --test 0.2
""")
md("## 4 · Baseline: the base model, zero-shot, same prompt\nGreedy decoding so the numbers are reproducible.")
code("""
import torch, time, json
from unsloth import FastLanguageModel, is_bf16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only
BASE = 'unsloth/Llama-3.2-1B-Instruct-bnb-4bit'
MAX_SEQ = 3072
def load():
    # Unsloth writes a tokenizer cache (with symlinks) into the working dir; Drive can't store symlinks and the
    # cache then breaks with "expected value at line 1 column 1". Load from local disk instead.
    here = os.getcwd(); os.makedirs('/content/_hf', exist_ok=True); os.chdir('/content/_hf')
    try:
        return FastLanguageModel.from_pretrained(model_name=BASE, max_seq_length=MAX_SEQ, load_in_4bit=True, dtype=None)
    finally:
        os.chdir(here)
test = [json.loads(l) for l in open('data/test.jsonl')]
train_rows = [json.loads(l) for l in open('data/train.jsonl')]
assert train_rows and test, 'No training data yet: check the output of steps 0 to 3 before going on.'

def predict(model, tok, rows, path):
    done = {json.loads(l)['id'] for l in open(path)} if os.path.exists(path) else set()
    FastLanguageModel.for_inference(model)
    with open(path, 'a') as f:
        for i, r in enumerate(rows):
            if r['id'] in done: continue
            ids = tok.apply_chat_template(r['messages'][:-1], add_generation_prompt=True, return_tensors='pt').to('cuda')
            t = time.time()
            out = model.generate(input_ids=ids, max_new_tokens=512, do_sample=False, use_cache=True, pad_token_id=tok.eos_token_id)
            txt = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            f.write(json.dumps({'id': r['id'], 'output': txt, 'seconds': round(time.time() - t, 2)}, ensure_ascii=False) + '\\n'); f.flush()
            if i % 10 == 0: print(i, r['id'], txt[:120].replace('\\n', ' '))

model, tok = load()
tok = get_chat_template(tok, chat_template='llama-3.1')
predict(model, tok, test, 'runs/base_preds.jsonl')
from src.metrics import score, table, load as jl
base_m = score(test, jl('runs/base_preds.jsonl'))[0]; print(base_m)
# the same loaded model is reused for training in step 5
""")
md("""
## 5 · QLoRA fine-tune
Base frozen in 4-bit NF4; rank-16 LoRA adapters on all linear layers; loss on the JSON answer only (prompt tokens masked to -100).
lr 2e-4 with warmup + linear decay, effective batch 8 (2 × 4 accumulation), 8-bit AdamW, 3 epochs.
""")
code("""
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
if 'model' not in globals():  # runtime restarted after step 4: load it again
    model, tok = load()
    tok = get_chat_template(tok, chat_template='llama-3.1')
FastLanguageModel.for_training(model)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0, bias='none',
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    use_gradient_checkpointing='unsloth', random_state=3407)
model.print_trainable_parameters()

texts = [tok.apply_chat_template(r['messages'], tokenize=False) for r in train_rows]
lens = [len(tok(t).input_ids) for t in texts]
keep = [t for t, n in zip(texts, lens) if n <= MAX_SEQ]
print(f'train examples: {len(texts)}, token length max {max(lens)}, median {sorted(lens)[len(lens)//2]}, dropped (> {MAX_SEQ}): {len(texts) - len(keep)}')
ds = Dataset.from_dict({'text': keep})

import inspect
cfg = dict(dataset_text_field='text', max_seq_length=MAX_SEQ, packing=False,
    per_device_train_batch_size=2, gradient_accumulation_steps=4, warmup_steps=5, num_train_epochs=3,
    learning_rate=2e-4, lr_scheduler_type='linear', optim='adamw_8bit', weight_decay=0.01,
    fp16=not is_bf16_supported(), bf16=is_bf16_supported(), logging_steps=1, seed=3407,
    output_dir='runs/checkpoints', save_strategy='epoch', save_total_limit=1, report_to='none')
if 'max_seq_length' not in inspect.signature(SFTConfig.__init__).parameters:  # newer TRL renamed it
    cfg['max_length'] = cfg.pop('max_seq_length')
try:
    trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=ds, args=SFTConfig(**cfg))
except TypeError:  # TRL versions that only accept processing_class
    trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=ds, args=SFTConfig(**cfg))
trainer = train_on_responses_only(trainer,
    instruction_part='<|start_header_id|>user<|end_header_id|>\\n\\n',
    response_part='<|start_header_id|>assistant<|end_header_id|>\\n\\n')

# sanity check: only the assistant JSON should carry loss
lab = trainer.train_dataset[0]['labels']
print('tokens with loss in example 0:', sum(x != -100 for x in lab), 'of', len(lab))
print(tok.decode([t for t, l in zip(trainer.train_dataset[0]['input_ids'], lab) if l != -100])[:300])

stats = trainer.train()
print(stats)
json.dump(trainer.state.log_history, open('runs/train_log.json', 'w'))
model.save_pretrained('runs/lora_adapter'); tok.save_pretrained('runs/lora_adapter')
""")
code("""
import matplotlib.pyplot as plt
log = [x for x in trainer.state.log_history if 'loss' in x]
plt.figure(figsize=(7, 3)); plt.plot([x['step'] for x in log], [x['loss'] for x in log]); plt.xlabel('step'); plt.ylabel('training loss')
plt.title('If this drops to ~0, it is memorising: try 2 epochs'); plt.tight_layout(); plt.savefig('runs/train_loss.png'); plt.show()
""")
md("## 6 · Evaluate the fine-tuned model on the same held-out orders")
code("""
predict(model, tok, test, 'runs/ft_preds.jsonl')
ft_m = score(test, jl('runs/ft_preds.jsonl'))[0]
res = {'Base Llama 3.2 1B (zero-shot)': base_m, 'Fine-tuned (QLoRA)': ft_m}
md_table = table(res); print(md_table)
open('runs/results.md', 'w').write(f'Held-out test set: {len(test)} orders, {len({r["case"] for r in test})} cases. Labels are Gemini (teacher) labels, so these numbers measure agreement with the teacher.\\n\\n' + md_table + '\\n')
json.dump(res, open('runs/results.json', 'w'), indent=1)
""")
md("## 7 · Export for Case Intel (Ollama) — merged, 4-bit GGUF")
code("""
import glob, shutil
model.save_pretrained_gguf('/content/gguf', tok, quantization_method='q4_k_m')  # Unsloth writes the GGUF to /content/gguf_gguf
out = glob.glob('/content/gguf*/*.gguf')
src = os.path.dirname([f for f in out if 'Q4_K_M' in f][0])
shutil.copytree(src, 'runs/gguf', dirs_exist_ok=True)  # the .gguf plus the Modelfile Unsloth wrote
!ls -la runs/gguf
""")
md("## 8 · Try it on one order")
code("""
r = next(x for x in test if x['label']['directions'])  # an order that has real instructions in it
ids = tok.apply_chat_template(r['messages'][:-1], add_generation_prompt=True, return_tensors='pt').to('cuda')
out = model.generate(input_ids=ids, max_new_tokens=512, do_sample=False, pad_token_id=tok.eos_token_id)
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
print('\\nteacher label:', json.dumps(r['label'], ensure_ascii=False))
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
for c in nb["cells"]:
    c["source"] = [l + "\n" for l in c["source"].split("\n")]
    c["source"][-1] = c["source"][-1].rstrip("\n")
os.makedirs("notebooks", exist_ok=True)
json.dump(nb, open("notebooks/court_order_qlora.ipynb", "w"), indent=1, ensure_ascii=False)
print("wrote notebooks/court_order_qlora.ipynb with", len(cells), "cells")
