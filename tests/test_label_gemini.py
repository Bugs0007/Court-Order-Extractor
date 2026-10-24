"""Labelling logic with a fake Gemini client (no network, no key)."""
import json
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.schema import canonical  # noqa: E402
from tests.test_pipeline import GOOD, ORDER  # noqa: E402


def fake_genai(responses):
    calls = []

    class Models:
        def generate_content(self, model, contents, config):
            calls.append(contents)
            r = responses.pop(0)
            if isinstance(r, Exception):
                raise r
            return types.SimpleNamespace(text=r)

    class Client:
        def __init__(self, api_key):
            self.models = Models()

    class Cfg:
        def __init__(self, **kw):
            self.kw = kw

    genai = types.SimpleNamespace(Client=Client, types=types.SimpleNamespace(GenerateContentConfig=Cfg))
    return genai, calls


def run(tmp_path, monkeypatch, responses, n_orders):
    genai, calls = fake_genai(responses)
    google = types.ModuleType("google")
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", genai.types)
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    op, lp = tmp_path / "orders.jsonl", tmp_path / "labels.jsonl"
    op.write_text("\n".join(json.dumps({"id": f"o{i}", "text": ORDER}) for i in range(n_orders)))
    from src import label_gemini
    label_gemini.main([str(op), str(lp), "--sleep", "0"])
    return [json.loads(l) for l in open(lp)], calls


def test_valid_invalid_retry_and_resume(tmp_path, monkeypatch):
    bad = json.dumps(dict(GOOD, outcome="postponed"))
    rows, calls = run(tmp_path, monkeypatch, [canonical(GOOD), bad, canonical(GOOD), bad, bad, bad], 3)
    assert [r["status"] for r in rows] == ["ok", "ok", "rejected"]  # o1 fixed on retry, o2 rejected after 3 tries
    assert rows[0]["label"] == GOOD and "outcome" in rows[2]["problems"][0]
    assert len(calls) == 6 and "Court order:" in calls[0]
    rows2, calls2 = run(tmp_path, monkeypatch, [], 3)  # re-run: everything already recorded, no calls
    assert len(rows2) == 3 and calls2 == []


def test_errors_are_retried_and_quota_stops(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    quota = Exception("429 RESOURCE_EXHAUSTED")
    rows, calls = run(tmp_path, monkeypatch, [quota, quota, quota], 3)  # first order hits the daily quota
    assert rows == [] and len(calls) == 3  # stopped, nothing recorded, other orders untouched
    rows, calls = run(tmp_path, monkeypatch, [canonical(GOOD)] * 3, 3)  # next day: all three get labelled
    assert [r["status"] for r in rows] == ["ok", "ok", "ok"] and len(calls) == 3
