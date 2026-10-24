"""Step 0: fetch real, public court orders from the open "Indian High Court Judgments"
dataset (eCourts data, CC-BY-4.0, s3://indian-high-court-judgments). Default court:
High Court for the State of Telangana. No AWS account needed: plain HTTPS.

    python -m src.fetch_open_orders data/raw_orders --n 400 [--years 2025 2024]

Picks at most --per-case orders per case (CNR) so the case-level train/test split has
many cases, spreads picks across disposal types, skips large files (long judgments,
heavy scans), and saves them with Case Intel's naming: <CNR>_order_<n>_<YYYY-MM-DD>.pdf.
Writes data/raw_orders/manifest.csv (what came from where) and SOURCE.md (attribution).
Re-running tops up to --n; files already there are kept.
"""
import argparse
import csv
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://indian-high-court-judgments.s3.ap-south-1.amazonaws.com"
FILE_RE = re.compile(r"^(?P<cnr>[A-Z]{4}\d{12})_(?P<n>\d+)_(?P<date>\d{4}-\d{2}-\d{2})\.pdf$", re.I)
CNR_RE = re.compile(r"^[A-Z]{4}\d{12}$")
UA = {"User-Agent": "court-order-extractor/1.0 (research; open dataset)"}


def http(url, method="GET", timeout=60, tries=3):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, method=method, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.headers, (r.read() if method == "GET" else b"")
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None, None
            err = e
        except Exception as e:  # network hiccup: retry
            err = e
        time.sleep(2 * (t + 1))
    print(f"  giving up on {url}: {err}")
    return None, None


def load_meta(years, court, bench, cache):
    import pandas as pd
    os.makedirs(cache, exist_ok=True)
    frames = []
    for y in years:
        path = os.path.join(cache, f"meta_{court}_{bench}_{y}.parquet")
        if not os.path.exists(path):
            url = f"{BASE}/metadata/parquet/year={y}/court={court}/bench={bench}/metadata.parquet"
            print(f"downloading metadata {y} ...")
            _, body = http(url, timeout=300)
            if not body:
                print(f"  no metadata for {y}, skipping")
                continue
            open(path, "wb").write(body)
        cols = None
        try:
            import pyarrow.parquet as pq
            have = set(pq.read_schema(path).names)
            cols = [c for c in ["cnr", "pdf_link", "decision_date", "disposal_nature", "title", "pdf_exists", "pdf_pages", "size"] if c in have]
        except Exception:
            pass
        df = pd.read_parquet(path, columns=cols)
        df["year"] = y
        frames.append(df)
        print(f"  {y}: {len(df):,} records")
    if not frames:
        sys.exit("No metadata downloaded. Check your internet connection.")
    return pd.concat(frames, ignore_index=True)


def txt(v):
    try:
        import pandas as pd
        if v is None or pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v)


def pd_num(s):
    import pandas as pd
    return pd.to_numeric(s, errors="coerce")


def target_name(row):
    base = row["_base"] if "_base" in row else os.path.basename(str(row["pdf_link"]).split("#")[0])
    m = FILE_RE.match(base)
    if m:
        return base, f"{m['cnr'].upper()}_order_{int(m['n'])}_{m['date']}.pdf"
    d = str(row.get("decision_date") or "")[:10]
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
        return base, f"{row['cnr']}_order_1_{d}.pdf"
    return base, None


def choose(df, n, per_case=1, max_pages=8, seed=7, log=print):
    """Candidate rows: valid CNR and file name, short enough, <= per_case per CNR,
    round-robin across disposal types so rare outcomes still appear.
    Optional filters (pdf_exists, pdf_pages) are skipped if they would drop everything."""
    base = df["pdf_link"].astype(str).str.split("#").str[0].str.rsplit("/", n=1).str[-1]
    from_name = base.str.extract(r"^([A-Za-z]{4}\d{12})_\d+_\d{4}-\d{2}-\d{2}\.pdf$", expand=False).str.upper()
    col = df["cnr"].astype("string").str.strip().str.upper() if "cnr" in df else None
    if col is not None:
        col = col.where(col.str.match(CNR_RE.pattern, na=False))
        cnr = from_name.fillna(col)
    else:
        cnr = from_name
    df = df.assign(cnr=cnr, _base=base)
    log(f"  {len(df):,} records; {df['cnr'].notna().sum():,} with a usable CNR")
    df = df[df["cnr"].notna() & df["_base"].str.lower().str.endswith(".pdf")]
    log(f"  {len(df):,} with a PDF file name")

    def soft(mask, label):
        kept = df[mask.fillna(True).astype(bool)]
        if len(kept) == 0:
            log(f"  {label}: would drop all {len(df):,}, ignoring this filter")
            return df
        log(f"  {label}: {len(kept):,} left")
        return kept
    if "pdf_exists" in df:
        df = soft(df["pdf_exists"].astype("boolean").ne(False), "pdf_exists")
    if "pdf_pages" in df:
        pages = pd_num(df["pdf_pages"])
        df = soft(pages.le(max_pages) | pages.isna(), f"<= {max_pages} pages")
    df = df.sample(frac=1, random_state=seed)
    df = df.groupby("cnr", sort=False).head(per_case)
    log(f"  {len(df):,} after keeping {per_case} order(s) per case")
    key = df["disposal_nature"].fillna("").astype(str).str.strip().str.upper() if "disposal_nature" in df else ""
    groups = [g for _, g in df.assign(_k=key).groupby("_k", sort=True)]
    random.Random(seed).shuffle(groups)
    out, idx = [], [0] * len(groups)
    while len(out) < n and any(i < len(g) for i, g in zip(idx, groups)):
        for gi, g in enumerate(groups):
            if idx[gi] < len(g):
                out.append(g.iloc[idx[gi]])
                idx[gi] += 1
                if len(out) >= n:
                    break
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="data/raw_orders")
    ap.add_argument("--n", type=int, default=400, help="target number of PDFs in the folder from this source")
    ap.add_argument("--years", type=int, nargs="+", default=[2025, 2024])
    ap.add_argument("--court", default="36_29", help="court code (36_29 = Telangana HC; see the dataset's high_courts.csv)")
    ap.add_argument("--bench", default="taphc")
    ap.add_argument("--per-case", type=int, default=1)
    ap.add_argument("--max-mb", type=float, default=1.0, help="skip PDFs larger than this (long judgments / heavy scans)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache", default=None, help="where to keep the metadata parquet (default: <out>/../_meta)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    cache = a.cache or os.path.join(os.path.dirname(os.path.abspath(a.out)), "_meta")
    manifest = os.path.join(a.out, "manifest.csv")
    have = {}
    if os.path.exists(manifest):
        have = {r["file"]: r for r in csv.DictReader(open(manifest, encoding="utf-8")) if os.path.exists(os.path.join(a.out, r["file"]))}
    need = a.n - len(have)
    if need <= 0:
        print(f"{len(have)} open-dataset orders already in {a.out}; nothing to do.")
        return 0

    df = load_meta(a.years, a.court, a.bench, cache)
    cands = [r for r in choose(df, a.n * 3, a.per_case, seed=a.seed) if target_name(r)[1] and target_name(r)[1] not in have]
    have_cnrs = {f.split("_order_")[0] for f in have}
    cands = [r for r in cands if r["cnr"] not in have_cnrs or a.per_case > 1]
    print(f"{len(df):,} records -> {len(cands):,} candidates; fetching {need} more (max {a.max_mb} MB each)")

    rows, skipped = list(have.values()), {"too_big": 0, "missing": 0, "not_pdf": 0}
    lim = int(a.max_mb * 1024 * 1024)

    def fetch(r):
        src, dst = target_name(r)
        url = f"{BASE}/data/pdf/year={r['year']}/court={a.court}/bench={a.bench}/{src}"
        hdr, _ = http(url, method="HEAD", timeout=30)
        if hdr is None:
            return "missing", None
        if int(hdr.get("Content-Length") or 0) > lim:
            return "too_big", None
        _, body = http(url, timeout=120)
        if not body:
            return "missing", None
        if not body.startswith(b"%PDF"):
            return "not_pdf", None
        open(os.path.join(a.out, dst), "wb").write(body)
        return "ok", {"file": dst, "cnr": r["cnr"], "decision_date": txt(r.get("decision_date"))[:10],
                      "disposal_nature": txt(r.get("disposal_nature")), "title": txt(r.get("title"))[:120],
                      "bytes": len(body), "source_url": url}

    got, i = 0, 0
    with ThreadPoolExecutor(a.workers) as ex:
        while got < need and i < len(cands):
            batch = cands[i: i + a.workers * 4]
            i += len(batch)
            for status, row in ex.map(fetch, batch):
                if status == "ok" and got < need:
                    rows.append(row)
                    got += 1
                elif status == "ok":  # batch overshot the target: drop the extra file
                    os.remove(os.path.join(a.out, row["file"]))
                elif status != "ok":
                    skipped[status] += 1
            print(f"  {len(rows)}/{a.n} in folder (skipped: {skipped})", flush=True)

    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "cnr", "decision_date", "disposal_nature", "title", "bytes", "source_url"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out, "SOURCE.md"), "w", encoding="utf-8") as f:
        f.write("Orders in manifest.csv come from the open dataset \"Indian High Court Judgments\" "
                "(https://registry.opendata.aws/indian-high-court-judgments/, "
                "code: https://github.com/vanga/indian-high-court-judgments), sourced from eCourts, "
                "licensed CC-BY-4.0. Court: " + a.court + ", bench: " + a.bench + ". Files renamed to "
                "<CNR>_order_<n>_<date>.pdf; contents unchanged.\n")
    from collections import Counter
    print(f"done: {len(rows)} PDFs, {len({r['cnr'] for r in rows})} distinct cases -> {a.out}")
    print("disposal types:", dict(Counter(r["disposal_nature"] or "(blank)" for r in rows).most_common(10)))
    return 0 if len(rows) >= a.n else 1


if __name__ == "__main__":
    sys.exit(main())
