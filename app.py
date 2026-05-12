"""
CoE Framework Analysis Tool — Flask backend
Run:  python app.py
Open: http://127.0.0.1:8501
"""

from __future__ import annotations

import gc
import os
import re
import sys
import traceback
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.resolve()
UPLOAD_DIR    = BASE_DIR / "uploads"
OUTPUT_DIR    = BASE_DIR / "outputs"
TEMPLATE_FILE = BASE_DIR / "CoE_Framework_Analysis_Templates.xlsm"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(BASE_DIR))

MIN_OUTPUT_BYTES = 5_000

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


def _safe_fn(name: str) -> str:
      name = (name or "").strip()
      name = re.sub(r'[^a-zA-Z0-9 \-_]', '', name)
      name = re.sub(r'\s+', ' ', name).strip()
      return name or "UNKNOWN_ACCOUNT"


def run_full_analysis(input_path: str) -> dict:
      from reader_databricks import load_databricks_export
      from rules_engine import evaluate_all
      from writer_framework import write_results_to_template

    if not TEMPLATE_FILE.exists():
              raise FileNotFoundError(f"Template not found: {TEMPLATE_FILE}")

    ctx = load_databricks_export(input_path)
    hash_name = getattr(ctx, "hash_name", "") or getattr(ctx, "account_name", "") or "UNKNOWN_ACCOUNT"
    safe_hash = _safe_fn(hash_name)

    results = evaluate_all(ctx)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dname = f"{safe_hash} - Framework Analysis - {ts}.xlsm"
    dpath = OUTPUT_DIR / dname

    write_results_to_template(
              template_path=str(TEMPLATE_FILE),
              output_path=str(dpath),
              results=results,
              ctx=ctx,
              sheet_analysis="Framework_Analysis",
              sheet_reference="Framework_Reference",
    )

    size = dpath.stat().st_size if dpath.exists() else 0
    if not dpath.exists() or size < MIN_OUTPUT_BYTES:
              raise RuntimeError(f"Output file missing or too small ({size} bytes).")

    window_str    = getattr(ctx, "window_str", "")
    ref_date      = str(getattr(ctx, "ref_date", "") or "")
    downloaded    = str(getattr(ctx, "downloaded_dt", "") or "")
    flag_ids      = [c for c, r in results.items() if r.status == "FLAG"]
    partial_ids   = [c for c, r in results.items() if r.status == "PARTIAL"]
    ok_count      = sum(1 for r in results.values() if r.status == "OK")
    flag_count    = sum(1 for r in results.values() if r.status == "FLAG")
    partial_count = sum(1 for r in results.values() if r.status == "PARTIAL")

    del ctx, results
    gc.collect()

    return {
              "download_filename": dname,
              "account":           hash_name,
              "window":            window_str,
              "ref_date":          ref_date,
              "downloaded":        downloaded,
              "ok":                ok_count,
              "flag":              flag_count,
              "partial":           partial_count,
              "flag_ids":          flag_ids,
              "partial_ids":       par
