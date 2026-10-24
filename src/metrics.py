"""Step 5: score predictions against the held-out labels.

    python -m src.metrics data/test.jsonl runs/base_preds.jsonl runs/ft_preds.jsonl

Metrics match what Case Intel needs from an order, not generic text overlap:
  json_valid       raw output is exactly one JSON object (anything else breaks the pipeline)
  schema_valid     ... and it has the right keys, enums and date formats
  outcome_acc      outcome enum matches (drives the hearing card status)
  next_date_acc    next_hearing_date matches, null == null counts (drives the calendar)
  order_date_acc, case_no_acc
  directions P/R/F1  a predicted direction counts if the party matches and the action
                   shares >= half its words with a gold direction for that party
  sec_per_order    mean generation time
Unparseable outputs count as wrong on every field (not skipped).
"""
import json
import re
import sys

from .schema import parse_model_output, validate

STOP = {"the", "a", "an", "to", "of", "and", "in", "on", "for", "by", "with", "is", "be", "its", "their", "his", "her"}


def words(s):
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in STOP}


def case_norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower()) or None


def dir_match(p, g):
    if p.get("party") != g.get("party"):
        return False
    a, b = words(p.get("action")), words(g.get("action"))
    if not a or not b:
        return False
    return len(a & b) / min(len(a), len(b)) >= 0.5


def match_directions(pred, gold):
    """Greedy one-to-one matching; returns true positives."""
    used, tp = set(), 0
    for p in pred or []:
        if not isinstance(p, dict):
            continue
        for j, g in enumerate(gold or []):
            if j not in used and dir_match(p, g):
                used.add(j)
                tp += 1
                break
    return tp


def score(gold_rows, pred_rows):
    preds = {r["id"]: r for r in pred_rows}
    n = len(gold_rows)
    c = dict(json_valid=0, parseable=0, schema_valid=0, outcome=0, next_date=0, order_date=0, case_no=0)
    tp = n_pred = n_gold = 0
    secs, per = [], []
    for g in gold_rows:
        gl = g["label"]
        pr = preds.get(g["id"], {})
        out = pr.get("output", "")
        if "seconds" in pr:
            secs.append(pr["seconds"])
        obj, strict = parse_model_output(out)
        n_gold += len(gl["directions"])
        row = {"id": g["id"], "json_valid": strict, "outcome_ok": False, "next_date_ok": False}
        if obj is not None:
            c["parseable"] += 1
            c["json_valid"] += strict
            c["schema_valid"] += strict and not validate(obj)
            row["outcome_ok"] = obj.get("outcome") == gl["outcome"]
            row["next_date_ok"] = obj.get("next_hearing_date") == gl["next_hearing_date"]
            c["outcome"] += row["outcome_ok"]
            c["next_date"] += row["next_date_ok"]
            c["order_date"] += obj.get("order_date") == gl["order_date"]
            c["case_no"] += case_norm(obj.get("case_number")) == case_norm(gl["case_number"])
            pd = obj.get("directions") if isinstance(obj.get("directions"), list) else []
            n_pred += len(pd)
            tp += match_directions(pd, gl["directions"])
        per.append(row)
    pct = lambda k: round(100 * c[k] / n, 1) if n else 0.0
    P = tp / n_pred if n_pred else 0.0
    R = tp / n_gold if n_gold else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return {
        "n": n, "json_valid": pct("json_valid"), "schema_valid": pct("schema_valid"), "parseable": pct("parseable"),
        "outcome_acc": pct("outcome"), "next_date_acc": pct("next_date"), "order_date_acc": pct("order_date"), "case_no_acc": pct("case_no"),
        "directions_p": round(100 * P, 1), "directions_r": round(100 * R, 1), "directions_f1": round(100 * F, 1),
        "sec_per_order": round(sum(secs) / len(secs), 2) if secs else None,
    }, per


LABELS = [("json_valid", "Valid JSON %"), ("schema_valid", "Schema-valid %"), ("outcome_acc", "Outcome accuracy %"),
          ("next_date_acc", "Next-hearing-date accuracy %"), ("order_date_acc", "Order-date accuracy %"), ("case_no_acc", "Case-number accuracy %"),
          ("directions_f1", "Directions F1 %"), ("sec_per_order", "Seconds per order")]


def table(results):
    """results: {"Base model": metrics, "Fine-tuned": metrics} -> markdown table"""
    names = list(results)
    out = ["| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for k, lab in LABELS:
        out.append(f"| {lab} | " + " | ".join("" if results[n].get(k) is None else str(results[n][k]) for n in names) + " |")
    return "\n".join(out)


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


if __name__ == "__main__":
    gold = load(sys.argv[1])
    res = {}
    for p in sys.argv[2:]:
        res[p.rsplit("/", 1)[-1].replace("_preds.jsonl", "")] = score(gold, load(p))[0]
    print(table(res))
