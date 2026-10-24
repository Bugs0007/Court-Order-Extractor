"""Step 1: court-order PDFs -> clean text, one JSON line per unique order.

    python -m src.extract_text data/raw_orders data/orders.jsonl [--ocr]

Case Intel stores orders as <CNR>_order_<n>_<YYYY-MM-DD>.pdf, so the CNR and order
date come from the filename when present. Scanned PDFs have no text layer; with
--ocr they go through Tesseract, otherwise they are skipped and listed.
"""
import argparse
import hashlib
import json
import os
import re
import sys

NAME_RE = re.compile(r"^(?P<cnr>[A-Z]{4}\d{12})_order_(?P<seq>\d+)_(?P<date>\d{4}-\d{2}-\d{2})", re.I)
MIN_CHARS = 200


def clean(text):
    text = text.replace("\x0c", "\n").replace(" ", " ")
    lines = []
    for ln in text.splitlines():
        ln = re.sub(r"[ \t]+", " ", ln).strip()
        if re.fullmatch(r"(page\s*)?\d+(\s*(of|/)\s*\d+)?", ln, re.I):
            continue  # bare page numbers
        lines.append(ln)
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def pdf_text(path):
    from pypdf import PdfReader
    r = PdfReader(path)
    return "\n".join((p.extract_text() or "") for p in r.pages)


def ocr_text(path):
    import pytesseract
    from pdf2image import convert_from_path
    return "\n".join(pytesseract.image_to_string(im, lang="eng") for im in convert_from_path(path, dpi=300))


def meta_from_name(fn):
    m = NAME_RE.match(fn)
    if not m:
        return {"cnr": None, "order_seq": None, "file_order_date": None}
    return {"cnr": m["cnr"].upper(), "order_seq": int(m["seq"]), "file_order_date": m["date"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--ocr", action="store_true", help="OCR PDFs that have no text layer")
    a = ap.parse_args(argv)
    files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(a.src) for f in fs if f.lower().endswith((".pdf", ".txt")))
    seen, rows, skipped = set(), [], []
    for path in files:
        fn = os.path.basename(path)
        used_ocr = False
        try:
            if fn.lower().endswith(".txt"):
                raw = open(path, encoding="utf-8", errors="replace").read()
            else:
                raw = pdf_text(path)
                if len(clean(raw)) < MIN_CHARS and a.ocr:
                    raw, used_ocr = ocr_text(path), True
        except Exception as e:  # corrupt / encrypted PDFs
            skipped.append((fn, f"error: {e}"))
            continue
        text = clean(raw)
        if len(text) < MIN_CHARS:
            skipped.append((fn, "no text layer (scanned?)" if not a.ocr else "too little text after OCR"))
            continue
        h = hashlib.sha256(re.sub(r"\s+", " ", text).lower().encode()).hexdigest()
        if h in seen:
            skipped.append((fn, "duplicate"))
            continue
        seen.add(h)
        rows.append({"id": os.path.splitext(fn)[0], "sha256": h, **meta_from_name(fn), "ocr": used_ocr, "chars": len(text), "text": text})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(files)} files -> {len(rows)} unique orders written to {a.out}; skipped {len(skipped)}")
    for fn, why in skipped[:30]:
        print(f"  skipped {fn}: {why}")
    if len(skipped) > 30:
        print(f"  ... {len(skipped) - 30} more")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
