Held-out test set: 78 orders, 78 cases. Labels are Gemini (gemini-3.1-flash-lite) labels, so these numbers measure agreement with the teacher.

| Metric | Base Llama 3.2 1B (zero-shot) | Fine-tuned (QLoRA) |
|---|---|---|
| Valid JSON % | 17.9 | 98.7 |
| Schema-valid % | 0.0 | 97.4 |
| Outcome accuracy % | 14.1 | 92.3 |
| Next-hearing-date accuracy % | 0.0 | 98.7 |
| Order-date accuracy % | 6.4 | 92.3 |
| Case-number accuracy % | 2.6 | 78.2 |
| Directions F1 % | 0.0 | 66.7 |
| Seconds per order | 3.82 | 3.1 |

Copied from the Colab output of step 6. The full runs/ folder (preds, train_log.json, train_loss.png, LoRA adapter) is in Google Drive: MyDrive/Fine-Tuning-CaseIntel/runs.
