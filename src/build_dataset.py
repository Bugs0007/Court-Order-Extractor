"""Step 3: join orders + accepted labels, split by CASE, write chat-format JSONL.

    python -m src.build_dataset data/orders.jsonl data/labels.jsonl data/ [--test 0.2]

Split is by case (CNR) when the filename has one, else by document, so no order of
a test case is ever seen in training (two orders of the same case share a lot of
text, which would leak). Optional data/review.jsonl holds hand corrections:
{"id": ..., "label": {...}} or {"id": ..., "drop": true}; they override Gemini.
"""
import argparse
import hashlib
import json
import os
import sys

from .prompt import messages
from .schema import canonical, validate


def bucket(key, test_frac):
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 10000
    return "test" if h < test_frac * 10000 else "train"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("orders")
    ap.add_argument("labels")
    ap.add_argument("outdir")
    ap.add_argument("--test", type=float, default=0.2)
    ap.add_argument("--review", default=None, help="hand corrections JSONL (default: <outdir>/review.jsonl if present)")
    a = ap.parse_args(argv)
    orders = {o["id"]: o for o in map(json.loads, open(a.orders, encoding="utf-8"))}
    labels = {}
    for r in map(json.loads, open(a.labels, encoding="utf-8")):
        if r.get("status") == "ok":
            labels[r["id"]] = r["label"]
    rv = a.review or os.path.join(a.outdir, "review.jsonl")
    n_fix = n_drop = 0
    if os.path.exists(rv):
        for r in map(json.loads, open(rv, encoding="utf-8")):
            if r.get("drop"):
                labels.pop(r["id"], None)
                n_drop += 1
            elif "label" in r:
                assert not validate(r["label"]), f"review label for {r['id']} invalid: {validate(r['label'])}"
                labels[r["id"]] = r["label"]
                n_fix += 1
    split = {"train": [], "test": []}
    for oid, lab in sorted(labels.items()):
        o = orders.get(oid)
        if not o:
            continue
        key = o.get("cnr") or oid
        s = bucket(key, a.test)
        split[s].append({"id": oid, "case": key, "messages": messages(o["text"], canonical(lab)), "label": lab})
    if not split["test"] and len(split["train"]) > 4:  # tiny sets: force at least one test case
        cases = sorted({r["case"] for r in split["train"]})
        move = set(cases[: max(1, int(len(cases) * a.test))])
        split["test"] = [r for r in split["train"] if r["case"] in move]
        split["train"] = [r for r in split["train"] if r["case"] not in move]
    os.makedirs(a.outdir, exist_ok=True)
    for s, rows in split.items():
        with open(os.path.join(a.outdir, f"{s}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tr, te = split["train"], split["test"]
    overlap = {r["case"] for r in tr} & {r["case"] for r in te}
    print(f"labels used: {len(labels)} (review: {n_fix} fixed, {n_drop} dropped)")
    print(f"train: {len(tr)} orders / {len({r['case'] for r in tr})} cases;  test: {len(te)} orders / {len({r['case'] for r in te})} cases;  case overlap: {len(overlap)}")
    return 0 if tr and te and not overlap else 1


if __name__ == "__main__":
    sys.exit(main())
