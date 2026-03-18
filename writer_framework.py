# writer_framework.py
from __future__ import annotations

from typing import Dict, Tuple, Optional

from openpyxl import load_workbook

OK = "OK"
FLAG = "FLAG"
PARTIAL = "PARTIAL"

# C048 is the only manual control (Cleanup & Relaunch Cadence)
MANUAL_CONTROLS = {"C048"}


def _safe_str(x) -> str:
    return str(x).strip() if x is not None else ""


def _get_hash_name(ctx) -> str:
    if ctx is None:
        return "UNKNOWN ACCOUNT"
    v = _safe_str(getattr(ctx, "hash_name", ""))
    if v:
        return v
    v = _safe_str(getattr(ctx, "account_name", ""))
    return v or "UNKNOWN ACCOUNT"


def _get_tenant_account_line(ctx) -> str:
    if ctx is None:
        return ""
    tenant_id = _safe_str(getattr(ctx, "tenant_id", ""))
    account_id = _safe_str(getattr(ctx, "account_id", ""))
    parts = []
    if tenant_id:
        parts.append(f"Tenant ID: {tenant_id}")
    if account_id:
        parts.append(f"Account ID: {account_id}")
    return " | ".join(parts)


def _get_eval_window_line(ctx) -> str:
    if ctx is None:
        return "UNKNOWN WINDOW"
    v = _safe_str(getattr(ctx, "window_str", ""))
    return v or "UNKNOWN WINDOW"


def _get_downloaded_line(ctx) -> str:
    if ctx is None:
        return ""
    dt = getattr(ctx, "downloaded_dt", None)
    if dt is None:
        return ""
    try:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return _safe_str(dt)


def _detect_control_id_column(ws) -> Tuple[int, int]:
    """
    Returns (header_row, control_col).
    Tries to find a header cell containing both 'control' and 'id'.
    Falls back to (1, 1).
    """
    for r in range(1, 60):
        for c in range(1, 60):
            val = ws.cell(r, c).value
            if val and ("control" in str(val).lower()) and ("id" in str(val).lower()):
                return r, c
    return 1, 1


def write_results_to_template(
    template_path: str,
    output_path: str,
    results: dict,  # {"C001": ControlResult-like} or {"C001":{"status":"OK","note":"..."}}
    ctx=None,
    sheet_analysis: str = "Framework_Analysis",
    sheet_reference: str = "Framework_Reference",
) -> Dict[str, object]:
    wb = load_workbook(
        template_path,
        keep_vba=True,
        keep_links=True,
        data_only=False,
    )

    analysis_written = False
    rows_written = 0
    header_row = None
    control_col = None

    try:
        # -----------------------------
        # TAB 1 — Framework_Analysis
        # -----------------------------
        if sheet_analysis in wb.sheetnames and ctx is not None:
            ws_a = wb[sheet_analysis]
            ws_a["A1"].value = _get_hash_name(ctx)
            # A2 untouched
            ws_a["B3"].value = _get_tenant_account_line(ctx) or None
            ws_a["B4"].value = _get_eval_window_line(ctx) or None
            ws_a["B5"].value = _get_downloaded_line(ctx) or None
            analysis_written = True

        # -----------------------------
        # TAB 2 — Framework_Reference
        # -----------------------------
        if sheet_reference not in wb.sheetnames:
            raise ValueError(f"Sheet '{sheet_reference}' not found.")

        ws = wb[sheet_reference]
        header_row, control_col = _detect_control_id_column(ws)

        # Build ControlID -> row index once
        cid_to_row: Dict[str, int] = {}
        blanks_in_a_row = 0
        row = header_row + 1

        # safe cap to avoid infinite scans on weird templates
        max_row = ws.max_row if ws.max_row and ws.max_row > row else row + 500

        while row <= max_row:
            raw = _safe_str(ws.cell(row, control_col).value).upper()
            if not raw:
                blanks_in_a_row += 1
                if blanks_in_a_row >= 25:
                    break
                row += 1
                continue

            blanks_in_a_row = 0
            cid_to_row[raw] = row
            row += 1

        # Write results directly by iterating results dict
        for cid, result in results.items():
            cid_u = _safe_str(cid).upper()
            r = cid_to_row.get(cid_u)
            if not r:
                continue

            # Support both dict and ControlResult-like
            status = getattr(result, "status", None) if hasattr(result, "status") else None
            note = getattr(result, "note", None) if hasattr(result, "note") else None
            if status is None and isinstance(result, dict):
                status = result.get("status")
            if note is None and isinstance(result, dict):
                note = result.get("note")

            status_u = _safe_str(status).upper()
            note_s = _safe_str(note)

            # STATUS (Column D)
            ws[f"D{r}"].value = status_u if status_u else None

            # Notes/Intent (Column N)
            if cid_u in MANUAL_CONTROLS:
                final_note = "MANUAL CHECK REQUIRED"
            elif status_u in {FLAG, PARTIAL} and note_s:
                final_note = note_s
            else:
                final_note = ""

            ws[f"N{r}"].value = final_note if final_note else None
            rows_written += 1

        wb.save(output_path)

    finally:
        try:
            wb.close()
        except Exception:
            pass

    return {
        "output_path": output_path,
        "analysis_written": analysis_written,
        "reference_rows_written": rows_written,
        "header_row": header_row,
        "control_col": control_col,
    }
