"""Tests for everything that runs without a GPU. Uses made-up orders only.
    python -m pytest -q
"""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import build_dataset, extract_text, metrics  # noqa: E402
from src.prompt import MAX_CHARS, SYSTEM, clip, messages  # noqa: E402
from src.schema import canonical, parse_model_output, validate  # noqa: E402

GOOD = {"case_number": "WP No. 412 of 2024", "order_date": "2025-11-03", "next_hearing_date": "2025-11-21", "outcome": "adjourned",
        "directions": [{"party": "respondent", "action": "file counter affidavit", "deadline": "within 4 weeks"}],
        "summary": "Adjourned to 21 November; the respondent must file a counter affidavit within four weeks."}

ORDER = """IN THE HIGH COURT FOR THE STATE OF TELANGANA AT HYDERABAD
WRIT PETITION No. 412 of 2024
Between: A. Ravi Kumar ... Petitioner  And: State of Telangana ... Respondent
Date: 03-11-2025
Heard learned counsel for the petitioner. Learned Government Pleader seeks time to file counter.
The respondent shall file counter affidavit within four weeks. List on 21.11.2025.
""" * 2  # long enough to pass the minimum-length filter (fake text, for tests only)


def write(p, s):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w", encoding="utf-8").write(s)


def make_pdf(path, text):
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    for ln in text.splitlines():
        pdf.multi_cell(0, 5, ln or " ", new_x="LMARGIN", new_y="NEXT")
    pdf.output(path)


# ---------------- schema ----------------
def test_schema_accepts_good():
    assert validate(GOOD) == []


@pytest.mark.parametrize("patch,expect", [
    ({"outcome": "postponed"}, "outcome"), ({"next_hearing_date": "21.11.2025"}, "next_hearing_date"),
    ({"order_date": "2025-02-30"}, "order_date"), ({"directions": [{"party": "lawyer", "action": "x", "deadline": None}]}, "party"),
    ({"extra": 1}, "unexpected"), ({"summary": ""}, "summary"),
])
def test_schema_rejects(patch, expect):
    bad = dict(GOOD, **patch)
    assert any(expect in p for p in validate(bad))


def test_schema_missing_key():
    bad = dict(GOOD)
    del bad["directions"]
    assert any("missing" in p for p in validate(bad))


def test_parse_model_output():
    s = canonical(GOOD)
    assert parse_model_output(s) == (json.loads(s), True)
    obj, strict = parse_model_output("Sure! ```json\n" + s + "\n```")
    assert obj == json.loads(s) and strict is False
    assert parse_model_output("no json here") == (None, False)
    assert parse_model_output('{"outcome": "adjourned"') == (None, False)  # truncated


# ---------------- prompt ----------------
def test_clip_keeps_head_and_tail():
    t = "HEAD " + "x" * 20000 + " TAIL"
    c = clip(t)
    assert len(c) <= MAX_CHARS + 10 and c.startswith("HEAD") and c.endswith("TAIL") and "[...]" in c
    assert clip("short") == "short"


def test_messages_shape():
    m = messages("order", canonical(GOOD))
    assert [x["role"] for x in m] == ["system", "user", "assistant"] and m[0]["content"] == SYSTEM
    assert json.loads(m[2]["content"]) == GOOD


# ---------------- extraction ----------------
def test_extract(tmp_path):
    raw = tmp_path / "raw"
    os.makedirs(raw)
    make_pdf(str(raw / "TSHC010011232024_order_3_2025-11-03.pdf"), ORDER)
    write(str(raw / "copy.txt"), ORDER)  # same text -> duplicate
    write(str(raw / "tiny.txt"), "List on 21.11.2025.")  # too short
    write(str(raw / "TSHC010022222024_order_1_2025-10-01.txt"), ORDER.replace("412", "999"))
    out = tmp_path / "orders.jsonl"
    rc = extract_text.main([str(raw), str(out)])
    rows = [json.loads(l) for l in open(out)]
    assert rc == 0 and len(rows) == 2
    pdf_row = [r for r in rows if r["id"].startswith("TSHC01001123")][0]
    assert pdf_row["cnr"] == "TSHC010011232024" and pdf_row["order_seq"] == 3 and pdf_row["file_order_date"] == "2025-11-03"
    assert "counter affidavit" in pdf_row["text"]


# ---------------- dataset split ----------------
def test_build_dataset_splits_by_case(tmp_path):
    orders, labels = [], []
    for c in range(20):  # 20 cases x 3 orders each
        for k in range(3):
            oid = f"TSHC01{c:010d}_order_{k}_2025-01-0{k + 1}"
            orders.append({"id": oid, "cnr": f"TSHC01{c:010d}", "text": ORDER + str(c * 10 + k)})
            labels.append({"id": oid, "status": "ok", "label": GOOD})
    labels.append({"id": "rej", "status": "rejected"})
    op, lp = tmp_path / "o.jsonl", tmp_path / "l.jsonl"
    op.write_text("\n".join(map(json.dumps, orders)))
    lp.write_text("\n".join(map(json.dumps, labels)))
    write(str(tmp_path / "out" / "review.jsonl"), json.dumps({"id": orders[0]["id"], "drop": True}))
    rc = build_dataset.main([str(op), str(lp), str(tmp_path / "out")])
    tr = [json.loads(l) for l in open(tmp_path / "out" / "train.jsonl")]
    te = [json.loads(l) for l in open(tmp_path / "out" / "test.jsonl")]
    assert rc == 0 and len(tr) + len(te) == 59 and te
    assert not ({r["case"] for r in tr} & {r["case"] for r in te})
    assert all(r["messages"][-1]["role"] == "assistant" for r in tr)


# ---------------- metrics ----------------
def test_metrics_perfect_and_broken():
    gold = [{"id": "a", "label": GOOD}, {"id": "b", "label": dict(GOOD, outcome="dismissed", next_hearing_date=None, directions=[])}]
    perfect = [{"id": "a", "output": canonical(GOOD), "seconds": 2.0}, {"id": "b", "output": canonical(gold[1]["label"]), "seconds": 4.0}]
    m, _ = metrics.score(gold, perfect)
    assert m["json_valid"] == 100 and m["outcome_acc"] == 100 and m["next_date_acc"] == 100 and m["directions_f1"] == 100 and m["sec_per_order"] == 3.0
    broken = [{"id": "a", "output": "Here you go: " + canonical(GOOD)}, {"id": "b", "output": "I cannot"}]
    m, _ = metrics.score(gold, broken)
    assert m["json_valid"] == 0 and m["parseable"] == 50 and m["outcome_acc"] == 50


def test_direction_matching():
    g = [{"party": "respondent", "action": "file counter affidavit", "deadline": None}]
    assert metrics.match_directions([{"party": "respondent", "action": "file the counter", "deadline": None}], g) == 1
    assert metrics.match_directions([{"party": "petitioner", "action": "file counter affidavit", "deadline": None}], g) == 0


def test_metrics_cli_table(tmp_path):
    gp, pp = tmp_path / "test.jsonl", tmp_path / "ft_preds.jsonl"
    gp.write_text(json.dumps({"id": "a", "label": GOOD}))
    pp.write_text(json.dumps({"id": "a", "output": canonical(GOOD), "seconds": 1.5}))
    out = subprocess.run([sys.executable, "-m", "src.metrics", str(gp), str(pp)], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    assert "| Outcome accuracy % | 100.0 |" in out
