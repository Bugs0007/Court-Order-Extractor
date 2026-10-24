"""fetch_open_orders against a fake local copy of the bucket (no internet)."""
import csv
import functools
import http.server
import os
import sys
import threading

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
from src import extract_text, fetch_open_orders as fo  # noqa: E402

PDF = b"%PDF-1.4\n% fake order\n"


@pytest.fixture
def bucket(tmp_path, monkeypatch):
    b = tmp_path / "bucket"
    meta = b / "metadata/parquet/year=2025/court=36_29/bench=taphc"
    pdfs = b / "data/pdf/year=2025/court=36_29/bench=taphc"
    meta.mkdir(parents=True)
    pdfs.mkdir(parents=True)
    rows = []
    for i in range(30):
        cnr = f"HBHC01{i:06d}2025"
        for k in (1, 2):  # two orders per case: only one should be picked
            name = f"{cnr}_{k}_2025-0{k}-1{i % 9}.pdf"
            rows.append({"cnr": cnr, "pdf_link": f"court/cnrorders/taphc/orders/2025/{name}#page=", "decision_date": f"2025-0{k}-1{i % 9}",
                         "disposal_nature": ["DISMISSED", "ALLOWED", None, "DISPOSED OF"][i % 4], "title": f"WP {i}", "pdf_exists": True})
            if i == 3:
                (pdfs / name).write_bytes(PDF + b"x" * (2 * 1024 * 1024))  # too big
            elif i == 4:
                (pdfs / name).write_bytes(b"<html>not a pdf</html>")
            elif i == 5:
                pass  # missing on the server
            else:
                (pdfs / name).write_bytes(PDF)
    rows.append({"cnr": "BADCNR", "pdf_link": "court/x.pdf", "decision_date": "2025-01-01", "disposal_nature": None, "title": "", "pdf_exists": True})
    pd.DataFrame(rows).to_parquet(meta / "metadata.parquet")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(b))
    handler.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(fo, "BASE", f"http://127.0.0.1:{srv.server_address[1]}")
    monkeypatch.setattr(fo.time, "sleep", lambda s: None)
    yield tmp_path
    srv.shutdown()


def test_fetch_picks_one_per_case_skips_bad_and_tops_up(bucket):
    out = bucket / "data" / "raw_orders"
    rc = fo.main([str(out), "--n", "12", "--years", "2025", "--workers", "3"])
    files = sorted(f for f in os.listdir(out) if f.endswith(".pdf"))
    man = list(csv.DictReader(open(out / "manifest.csv", encoding="utf-8")))
    assert rc == 0 and len(files) == 12 and len(man) == 12 and sorted(r["file"] for r in man) == files
    cnrs = [f.split("_order_")[0] for f in files]
    assert len(set(cnrs)) == 12  # one order per case
    bad = {f"HBHC01{i:06d}2025" for i in (3, 4, 5)}
    assert not bad & set(cnrs)
    assert all(extract_text.meta_from_name(f)["cnr"] for f in files)  # Case Intel naming parses
    assert (out / "SOURCE.md").read_text().count("CC-BY-4.0") == 1
    assert len({r["disposal_nature"] for r in man}) >= 3  # spread across outcomes
    rc = fo.main([str(out), "--n", "20", "--years", "2025", "--workers", "3"])  # top up, cached metadata
    files2 = [f for f in os.listdir(out) if f.endswith(".pdf")]
    assert rc == 0 and len(files2) == 20 and set(files) <= set(files2)
    assert len({f.split("_order_")[0] for f in files2}) == 20


def test_fetch_when_cnr_column_empty_and_pdf_exists_false(bucket):
    """Real-data shape seen in Colab: 0 candidates. CNR must come from the file name,
    and a pdf_exists flag that is False everywhere must not drop everything."""
    meta = bucket / "bucket/metadata/parquet/year=2025/court=36_29/bench=taphc/metadata.parquet"
    df = pd.read_parquet(meta)
    df["cnr"] = ""
    df["pdf_exists"] = False
    df["pdf_pages"] = None
    df.to_parquet(meta)
    out = bucket / "data" / "raw_orders"
    rc = fo.main([str(out), "--n", "10", "--years", "2025", "--workers", "2"])
    files = [f for f in os.listdir(out) if f.endswith(".pdf")]
    assert rc == 0 and len(files) == 10 and len({f.split("_order_")[0] for f in files}) == 10
