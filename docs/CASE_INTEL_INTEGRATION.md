# How this fits into Case Intel

## The problem it solves
Case Intel shows advocates what happened at each hearing: the outcome, the next date, and who must do what (petitioner vs respondent). Today the order-overview summary is produced by a hosted LLM call (Groq, OpenAI or similar). That has three costs:
- **Money per order**, which grows with every user and every order.
- **A paid API key** that production needs before summaries work at all.
- **Client documents leave the server.** Advocates care about confidentiality, and a model running on your own box keeps the order text inside your infrastructure.

This project trains a small model that does just that one job: order text in, structured JSON out.

## Where it plugs in
Case Intel already has the pieces:
- `CourtOrder` PDFs are stored and fetched per hearing.
- The order-overview summary is generated at ingest and cached.
- The AI service factory already supports an **Ollama** provider next to Groq and OpenAI.
- Party-aware display is applied at render time, from `Case.user_party_role`.

So the fine-tuned model slots in as one more provider behind that factory. No new feature, no schema change.

```
new order PDF arrives
  -> extract text (pypdf, OCR fallback)            [same as extract_text.py]
  -> build the prompt (SYSTEM + clip)              [prompt.py, copied into core/]
  -> local model via Ollama (GGUF from Step 7)
  -> validate JSON against the schema              [schema.py]
        valid   -> store on the order, cache it
        invalid -> fall back to the hosted LLM, mark which model answered
  -> render time: show "your side" vs "other side" from user_party_role
```

The JSON fields map straight onto what the product shows:

| Model output | Used for in Case Intel |
|---|---|
| `outcome` | status on the hearing card |
| `next_hearing_date` | calendar and reminders |
| `directions[].party` and `action` | "tasks for you" vs "tasks for the other side" |
| `summary` | the one-line overview on case details |

## How to implement it (steps)
1. Export the model (Step 7 of the notebook): `runs/gguf/Llama-3.2-1B-Instruct.Q4_K_M.gguf` (808 MB) plus the `Modelfile` Unsloth writes next to it.
2. On the server: install Ollama, run `ollama create court-order-extractor -f Modelfile`.
3. Copy `prompt.py` and `schema.py` into `core/` so the prompt and validator are identical to training.
4. Add `LocalOrderExtractor` to the AI service factory: call Ollama, `parse_model_output()`, `validate()`.
5. Routing rule: try local first; if the JSON is invalid or `outcome` is `other`, call the hosted model and record `extracted_by` on the order.
6. Run it in the existing background job that ingests orders, not in the web request, because CPU inference takes seconds.
7. Log valid-JSON rate and fallback rate per week. That is your real-world metric.

## Things to be straight about
- **Not integrated yet.** This repo is the trained model and its evaluation. Say "I built and evaluated it, and here is the integration design", unless you build step 4 first.
- **Server size.** A 1B 4-bit model needs about 1 GB of RAM plus working memory. The current t3.small (2 GB) is tight if it also runs gunicorn and Postgres connections; a t3.medium or a separate small box is safer. CPU generation will be several seconds per order, which is fine for background ingest.
- **Accuracy.** 92% outcome accuracy is against Gemini labels on 78 Telangana HC orders. Directions are at 66.7 F1. That is why the fallback exists, and why the summary shown to advocates should come with the existing order PDF link.
- **Other courts.** Trained only on Telangana High Court orders. District courts and other High Courts will need more labelled data.

## What to do next to make it real
1. Hand-label about 50 orders as a gold set (also fixes the "agreement with the teacher" caveat).
2. Add 300 to 500 more orders, with more directions-heavy ones, and retrain.
3. Build the provider and fallback (steps 4 and 5) behind a feature flag.
4. Report the real fallback rate after a week on the recruiter account.
