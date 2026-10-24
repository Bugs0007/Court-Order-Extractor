"""Step 2: distil labels from Gemini (the "teacher").

    GEMINI_API_KEY=... python -m src.label_gemini data/orders.jsonl data/labels.jsonl

Uses JSON-schema mode at temperature 0, so every label parses. Resumable: orders
already labelled are skipped, so you can stop and re-run. Each label is checked
with schema.validate(); failures are retried once, then recorded as rejected.
"""
import argparse
import json
import os
import sys
import time

from .prompt import SYSTEM, user_msg
from .schema import GEMINI_SCHEMA, canonical, validate


def load_done(path):
    done = {}
    if os.path.exists(path):
        for ln in open(path, encoding="utf-8"):
            r = json.loads(ln)
            done[r["id"]] = r
    # errors (network, quota) are retried on the next run; ok and rejected are final
    return {k: v for k, v in done.items() if v.get("status") != "error"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("orders")
    ap.add_argument("out")
    ap.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite"))
    ap.add_argument("--limit", type=int, default=0, help="label at most N new orders")
    ap.add_argument("--sleep", type=float, default=4.0, help="seconds between calls (free tier is rate-limited)")
    a = ap.parse_args(argv)

    from google import genai
    from google.genai import types

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit("Set GEMINI_API_KEY first (https://aistudio.google.com/apikey).")
    client = genai.Client(api_key=key)
    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM, temperature=0.0,
        response_mime_type="application/json", response_schema=GEMINI_SCHEMA,
    )
    done = load_done(a.out)
    orders = [json.loads(l) for l in open(a.orders, encoding="utf-8")]
    todo = [o for o in orders if o["id"] not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(orders)} orders, {len(done)} already labelled, labelling {len(todo)} with {a.model}")
    ok = bad = 0
    with open(a.out, "a", encoding="utf-8") as f:
        for i, o in enumerate(todo, 1):
            rec = {"id": o["id"], "teacher": a.model}
            quota = False
            for attempt in range(3):
                try:
                    r = client.models.generate_content(model=a.model, contents=user_msg(o["text"]), config=cfg)
                    obj = json.loads(r.text)
                    probs = validate(obj)
                    if not probs:
                        rec.update(status="ok", label=json.loads(canonical(obj)))
                        break
                    rec.update(status="rejected", problems=probs, raw=r.text)
                except Exception as e:
                    msg = str(e)
                    rec.update(status="error", error=msg[:300])
                    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                        quota = attempt == 2
                        time.sleep(30 * (attempt + 1))
                        continue
                time.sleep(a.sleep)
            if quota and rec["status"] == "error":
                print(f"Gemini quota exhausted at order {i}/{len(todo)}. Stopping; re-run this cell later (or tomorrow) and it resumes.")
                break
            ok += rec["status"] == "ok"
            bad += rec["status"] != "ok"
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(todo)}] {o['id']}: {rec['status']}")
            time.sleep(a.sleep)
    print(f"done: {ok} ok, {bad} not ok -> {a.out}")


if __name__ == "__main__":
    main()
