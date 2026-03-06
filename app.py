from __future__ import annotations
import os, sys, traceback, re
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, render_template, Response
from werkzeug.utils import secure_filename

BASE_DIR      = Path(__file__).parent.resolve()
UPLOAD_DIR    = BASE_DIR / "uploads"
OUTPUT_DIR    = BASE_DIR / "outputs"
TEMPLATE_FILE = BASE_DIR / "CoE_Framework_Analysis_Templates.xlsm"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(BASE_DIR))

from reader_databricks import load_databricks_export
from rules_engine import evaluate_all
from writer_framework import write_results_to_template

MIN_OUTPUT_BYTES = 5_000

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

def _safe_fn(name):
    name = (name or "").strip()
    name = re.sub(r'[^a-zA-Z0-9 \-_]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name or "UNKNOWN_ACCOUNT"

def run_full_analysis(input_path, daily_budget):
    if not TEMPLATE_FILE.exists():
        raise FileNotFoundError(f"Template not found: {TEMPLATE_FILE}")
    ctx = load_databricks_export(input_path)
    ctx.daily_budget = daily_budget
    hash_name = getattr(ctx, "hash_name", "") or getattr(ctx, "account_name", "") or "UNKNOWN_ACCOUNT"
    safe_hash = _safe_fn(hash_name)
    results = evaluate_all(ctx)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dname = f"{safe_hash} - Framework Analysis - {ts}.xlsm"
    dpath = OUTPUT_DIR / dname
    write_results_to_template(str(TEMPLATE_FILE), str(dpath), results, ctx, "Framework_Analysis", "Framework_Reference")
    size = dpath.stat().st_size if dpath.exists() else 0
    print(f"  Output written: {dpath} ({size} bytes)")
    if not dpath.exists() or size < MIN_OUTPUT_BYTES:
        raise RuntimeError(f"Output file missing or too small ({size} bytes).")
    return {
        "download_filename": dname,
        "account":     hash_name,
        "window":      getattr(ctx, "window_str", ""),
        "ref_date":    str(getattr(ctx, "ref_date", "") or ""),
        "downloaded":  str(getattr(ctx, "downloaded_dt", "") or ""),
        "daily_budget": daily_budget,
        "ok":      sum(1 for r in results.values() if r.status == "OK"),
        "flag":    sum(1 for r in results.values() if r.status == "FLAG"),
        "partial": sum(1 for r in results.values() if r.status == "PARTIAL"),
        "flag_ids":    [c for c, r in results.items() if r.status == "FLAG"],
        "partial_ids": [c for c, r in results.items() if r.status == "PARTIAL"],
    }

@app.route("/")
def index(): return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files: return jsonify({"error": "No file uploaded."}), 400
    f = request.files["file"]
    if not f.filename: return jsonify({"error": "No file selected."}), 400
    _, ext = os.path.splitext(f.filename.lower())
    if ext not in {".xlsx", ".xlsm"}: return jsonify({"error": "Only .xlsx or .xlsm accepted."}), 400
    try:
        budget = float(request.form.get("daily_budget", "0"))
        if budget <= 0: raise ValueError
    except ValueError:
        return jsonify({"error": "Enter a valid Daily Budget (number > 0)."}), 400
    ipath = str(UPLOAD_DIR / secure_filename(f.filename))
    f.save(ipath)
    try:
        info = run_full_analysis(ipath, budget)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Analysis failed: {e}"}), 500
    info["download_url"] = f"/download/{info['download_filename']}"
    return jsonify(info)

@app.route("/download/<path:filename>")
def download(filename):
    from urllib.parse import unquote
    filename = unquote(filename)
    p = OUTPUT_DIR / filename
    if not p.exists():
        xlsm_files = sorted(OUTPUT_DIR.glob("*.xlsm"), key=lambda f: f.stat().st_mtime, reverse=True)
        if xlsm_files:
            p = xlsm_files[0]
            filename = p.name
        else:
            return "No output files found.", 404
    print(f"  Serving download: {p} ({p.stat().st_size} bytes)")
    data = p.read_bytes()
    return Response(
        data,
        mimetype="application/vnd.ms-excel.sheet.macroEnabled.12",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(data)),
        }
    )

@app.route("/favicon.ico")
def favicon(): return "", 204

if __name__ == "__main__":
    print(f"\n  CoE Framework Analysis Tool")
    print(f"  Template exists: {TEMPLATE_FILE.exists()}")
    print(f"  Open → http://127.0.0.1:8501\n")
    app.run(host="127.0.0.1", port=8501, debug=True)
