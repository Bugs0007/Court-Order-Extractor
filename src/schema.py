"""The JSON the model must produce for one court order, plus a strict validator.

Kept flat and small on purpose: a 1B model learns a short, fixed format much more
reliably than a deep one, and enums make outcome accuracy easy to score.
"""
import json
import re
from datetime import date

OUTCOMES = [
    "adjourned",        # listed again, nothing decided
    "notice_issued",    # notice ordered to the other side
    "interim_order",    # stay / interim relief / interim directions
    "reserved",         # orders or judgment reserved
    "allowed",          # disposed in the petitioner's favour
    "dismissed",        # disposed against the petitioner (incl. for default)
    "disposed",         # disposed some other way (settled, closed, as infructuous)
    "withdrawn",        # withdrawn by the party
    "other",
]
PARTIES = ["petitioner", "respondent", "both", "court_office", "other"]

FIELDS = ["case_number", "order_date", "next_hearing_date", "outcome", "directions", "summary"]

# Gemini "response_schema" (OpenAPI subset). Order of properties is kept stable.
GEMINI_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "case_number": {"type": "STRING", "nullable": True, "description": "Case number as written in the order, e.g. 'WP No. 412 of 2024'. null if absent."},
        "order_date": {"type": "STRING", "nullable": True, "description": "Date the order was passed, YYYY-MM-DD. null if absent."},
        "next_hearing_date": {"type": "STRING", "nullable": True, "description": "Date the matter is listed next, YYYY-MM-DD. null if none (e.g. disposed) or not stated."},
        "outcome": {"type": "STRING", "enum": OUTCOMES, "description": "What happened at this hearing."},
        "directions": {
            "type": "ARRAY",
            "description": "Every concrete instruction in the order: who must do what, and by when.",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "party": {"type": "STRING", "enum": PARTIES},
                    "action": {"type": "STRING", "description": "Short imperative, e.g. 'file counter affidavit'."},
                    "deadline": {"type": "STRING", "nullable": True, "description": "As stated, e.g. 'within 4 weeks' or '2025-11-21'. null if none."},
                },
                "required": ["party", "action", "deadline"],
                "propertyOrdering": ["party", "action", "deadline"],
            },
        },
        "summary": {"type": "STRING", "description": "One plain-English sentence for the advocate, max 40 words."},
    },
    "required": FIELDS,
    "propertyOrdering": FIELDS,
}

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _valid_date(s):
    if s is None:
        return True
    if not isinstance(s, str) or not DATE_RE.match(s):
        return False
    try:
        date.fromisoformat(s)
        return True
    except ValueError:
        return False


def validate(obj):
    """Return a list of problems (empty list = valid)."""
    p = []
    if not isinstance(obj, dict):
        return ["not a JSON object"]
    missing = [f for f in FIELDS if f not in obj]
    if missing:
        p.append("missing: " + ",".join(missing))
    extra = [k for k in obj if k not in FIELDS]
    if extra:
        p.append("unexpected: " + ",".join(extra))
    for f in ("order_date", "next_hearing_date"):
        if f in obj and not _valid_date(obj[f]):
            p.append(f"{f} not YYYY-MM-DD: {obj[f]!r}")
    if "case_number" in obj and obj["case_number"] is not None and not isinstance(obj["case_number"], str):
        p.append("case_number not a string")
    if "outcome" in obj and obj["outcome"] not in OUTCOMES:
        p.append(f"outcome not in enum: {obj['outcome']!r}")
    d = obj.get("directions")
    if "directions" in obj:
        if not isinstance(d, list):
            p.append("directions not a list")
        else:
            for i, x in enumerate(d):
                if not isinstance(x, dict) or set(x) != {"party", "action", "deadline"}:
                    p.append(f"directions[{i}] wrong keys")
                    continue
                if x["party"] not in PARTIES:
                    p.append(f"directions[{i}].party not in enum: {x['party']!r}")
                if not isinstance(x["action"], str) or not x["action"].strip():
                    p.append(f"directions[{i}].action empty")
                if x["deadline"] is not None and not isinstance(x["deadline"], str):
                    p.append(f"directions[{i}].deadline not string/null")
    if "summary" in obj and (not isinstance(obj["summary"], str) or not obj["summary"].strip()):
        p.append("summary empty")
    return p


def parse_model_output(text):
    """Parse what a model generated. Tolerates a ```json fence or text around the
    object (so we can still score content), but `strict` tells you whether the raw
    output was a clean JSON object, which is what the pipeline actually needs."""
    t = text.strip()
    strict = None
    try:
        strict = json.loads(t)
    except Exception:
        pass
    if isinstance(strict, dict):
        return strict, True
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj, False
        except Exception:
            pass
    return None, False


def canonical(obj):
    """Stable, compact serialisation used as the training target."""
    out = {f: obj.get(f) for f in FIELDS}
    out["directions"] = [{"party": d["party"], "action": d["action"], "deadline": d.get("deadline")} for d in (out["directions"] or [])]
    return json.dumps(out, ensure_ascii=False, separators=(", ", ": "))
