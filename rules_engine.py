# rules_engine.py
from __future__ import annotations

import re  # ✅ FIX (3): needed for C038 regex
from datetime import timedelta
from typing import Callable, Dict, Optional, Tuple, List

import pandas as pd

import numpy as np

import config as cfg
from config import ControlResult
from reader_databricks import DatabricksContext, get_dataset


# =========================================================
# Locked policies / primitives
# =========================================================
def note_data_missing(tab: str, col: str) -> str:
    return f"DATA MISSING: {tab}.{col}"


def ok(note: str = "") -> ControlResult:
    return ControlResult(cfg.STATUS_OK, note or "")


def flag(note: str = "") -> ControlResult:
    """
    Hard policy: Every FLAG must include a deterministic note.
    If a control calls flag() without a note, raise so evaluate_all() records it as an EXCEPTION note.
    """
    note = (note or "").strip()
    if not note:
        note = "FLAG triggered but no deterministic note provided."
    return ControlResult(cfg.STATUS_FLAG, note)


def partial(note: str = "") -> ControlResult:
    return ControlResult(cfg.STATUS_PARTIAL, note or "")


def expected_tab_label(dataset_key: str) -> str:
    prefixes = getattr(cfg, "TAB_CANDIDATES", {}).get(dataset_key, [])
    return prefixes[0] if prefixes else dataset_key


# =========================================================
# Helpers (stable)
# =========================================================
def norm(s: str) -> str:
    return (
        str(s)
        .strip()
        .lower()
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("\t", " ")
        .replace(" ", "_")
    )


def excel_col_to_idx(col: str) -> int:
    col = col.upper().strip()
    if not col.isalpha():
        return -1
    n = 0
    for ch in col:
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n - 1


def get_col_by_letter(df: pd.DataFrame, letter: str) -> Optional[str]:
    idx = excel_col_to_idx(letter)
    if idx < 0 or idx >= len(df.columns):
        return None
    return df.columns[idx]


# =========================================================
# SPEED UPGRADE: cache normalized column maps for find_col()
# =========================================================
_COLMAP_CACHE: Dict[tuple, Dict[str, str]] = {}
_COLMAP_CACHE_MAX = 256


def _get_cols_norm_map(df: pd.DataFrame) -> Dict[str, str]:
    cols = list(df.columns)
    key = (id(df), tuple(cols))

    m = _COLMAP_CACHE.get(key)
    if m is not None:
        return m

    m = {norm(c): c for c in cols}

    if len(_COLMAP_CACHE) >= _COLMAP_CACHE_MAX:
        _COLMAP_CACHE.pop(next(iter(_COLMAP_CACHE)))

    _COLMAP_CACHE[key] = m
    return m


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_norm = _get_cols_norm_map(df)

    # exact matches
    for cand in candidates:
        cn = norm(cand)
        if cn in cols_norm:
            return cols_norm[cn]

    # substring matches
    for cand in candidates:
        cn = norm(cand)
        for k, orig in cols_norm.items():
            if cn in k:
                return orig

    return None


def as_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        if isinstance(x, str):
            s = x.strip().replace("%", "")
            if s == "":
                return None
            return float(s)
        return float(x)
    except Exception:
        return None


def as_int(x) -> Optional[int]:
    try:
        if pd.isna(x):
            return None
        if isinstance(x, str):
            s = x.strip()
            if s == "":
                return None
            return int(float(s))
        return int(float(x))
    except Exception:
        return None


def as_bool(x) -> Optional[bool]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"true", "t", "1", "yes", "y"}:
        return True
    if s in {"false", "f", "0", "no", "n"}:
        return False
    return None


def ds(
    ctx: DatabricksContext,
    dataset_key: str,
    fallback_tab_prefix: Optional[str] = None,
) -> Tuple[Optional[str], Optional[pd.DataFrame]]:
    """
    Dataset resolver with caching:
      - avoid re-reading/parsing the same dataset across multiple controls
      - cache is stored on ctx as _dataset_cache
    """
    # Create cache dict on ctx (safe monkey-patch)
    cache = getattr(ctx, "_dataset_cache", None)
    if cache is None:
        cache = {}
        setattr(ctx, "_dataset_cache", cache)

    # 1) try primary dataset_key from cache
    ck1 = ("ds", str(dataset_key))
    if ck1 in cache:
        sh, df = cache[ck1]
    else:
        sh, df = get_dataset(ctx, dataset_key)
        cache[ck1] = (sh, df)

    # 2) fallback if missing
    if (df is None) and fallback_tab_prefix:
        ck2 = ("ds", str(fallback_tab_prefix))
        if ck2 in cache:
            sh2, df2 = cache[ck2]
        else:
            sh2, df2 = get_dataset(ctx, fallback_tab_prefix)
            cache[ck2] = (sh2, df2)
        return sh2 or sh, df2

    return sh, df


def seller_param_row7(
    ctx: DatabricksContext,
    required_label: str,
    col_spec: str,
    no_data_flag: bool = True,
    dataset_key: str = "SELLER_PARAMS",
    fallback_tab_prefix: str = "40_Seller_Parameter_Insights_Da",
) -> Tuple[Optional[str], Optional[pd.DataFrame], Optional[str], Optional[ControlResult]]:
    """
    Fetch seller parameter value from row 7 ONLY (first data row after header).
    Returns (sheet_name, df, col_name, missing_result).
    """
    sh, df = ds(ctx, dataset_key, fallback_tab_prefix)
    if df is None or df.empty:
        return sh, df, None, (flag(note_data_missing(expected_tab_label(fallback_tab_prefix), required_label)) if no_data_flag else ok())

    col = find_col(df, [required_label] + col_spec.split("|"))
    if not col:
        return sh, df, None, flag(note_data_missing(sh or expected_tab_label(fallback_tab_prefix), required_label))
    if len(df.index) < 1:
        return sh, df, None, flag(note_data_missing(sh or expected_tab_label(fallback_tab_prefix), required_label))
    return sh, df, col, None


def any_rows(ctx: DatabricksContext, dataset_key: str, fallback_tab_prefix: Optional[str], no_data_ok: bool = True) -> ControlResult:
    sh, df = ds(ctx, dataset_key, fallback_tab_prefix)
    tab = sh or expected_tab_label(fallback_tab_prefix or dataset_key)

    if df is None or df.empty:
        return ok() if no_data_ok else flag(note_data_missing(tab, "Rows"))

    n = int(len(df.index))
    return flag(f"{tab}: {n} rows found (presence-based trigger).")


def _normalize_pct(v: float) -> float:
    # normalize decimals (0.16 -> 16)
    return v * 100.0 if 0 < v <= 1.0 else float(v)


# =========================================================
# Controls (C001–C054) — aligned to Mapping v8 (54 controls)
# =========================================================

# ---- C001/C002/C003: ACoS change governance ----
def _load_acos_changes(ctx: DatabricksContext) -> Tuple[Optional[str], Optional[pd.DataFrame], Optional[str], Optional[str]]:
    sh, df = ds(ctx, "ACOS_CHANGES", "24_Account_ACoS_Changes_History")
    if df is None or df.empty:
        return sh, df, None, None
    col_date = find_col(df, ["change_date", "change date", "changedate"])
    col_val = find_col(df, ["iacos_percent", "iacos percent", "iacos_percent_", "iacos", "iacos_percent"])
    return sh, df, col_date, col_val


def eval_C001(ctx: DatabricksContext) -> ControlResult:
    """
    FLAG: any gap between consecutive changes <14 days in last 90 days
    PARTIAL: any gap between consecutive changes <14 days in last 180 days (but NOT flagged in 90d)
    OK: otherwise (or <2 changes in window)
    No-data: OK
    """
    sh, df, col_date, _ = _load_acos_changes(ctx)
    if df is None or df.empty:
        return ok()
    if not ctx.ref_date or not col_date:
        return ok()  # Mapping v8: OK if no data; treat missing date as no-eval

    tmp = df.copy()
    tmp["_dt"] = pd.to_datetime(tmp[col_date], errors="coerce")
    tmp = tmp[tmp["_dt"].notna()].copy()
    if tmp.empty:
        return ok()

    def min_gap_days(window_days: int) -> Optional[int]:
        cutoff = pd.Timestamp(ctx.ref_date - timedelta(days=window_days))
        w = tmp[tmp["_dt"] >= cutoff].copy()
        if len(w.index) < 2:
            return None
        dts_sorted = w["_dt"].sort_values().reset_index(drop=True)
        gaps = dts_sorted.diff().dt.days.dropna()
        if gaps.empty:
            return None
        return int(gaps.min())

    g90 = min_gap_days(90)
    if g90 is not None and g90 < 14:
        return flag(f"Min gap between ACoS changes in last 90d = {g90} days (<14).")
    g180 = min_gap_days(180)
    if g180 is not None and g180 < 14:
        return partial(f"Min gap between ACoS changes in last 180d = {g180} days (<14).")
    return ok()


def eval_C002(ctx: DatabricksContext) -> ControlResult:
    """
    FLAG: change count in last 90 days > 5
    OK: otherwise
    No-data: OK
    """
    sh, df, col_date, _ = _load_acos_changes(ctx)
    if df is None or df.empty:
        return ok()
    if not ctx.ref_date or not col_date:
        return ok()

    dts = pd.to_datetime(df[col_date], errors="coerce").dropna()
    if dts.empty:
        return ok()
    cutoff_90 = pd.Timestamp(ctx.ref_date - timedelta(days=90))
    changes_90 = int((dts >= cutoff_90).sum())
    return flag(f"ACoS target changes in last 90d: {changes_90} (>5).") if changes_90 > 5 else ok()


def eval_C003(ctx: DatabricksContext) -> ControlResult:
    """
    FLAG: any relative magnitude <5% OR >25% between consecutive changes (in window)
    OK: otherwise (or <2 values)
    No-data: OK
    """
    sh, df, col_date, col_val = _load_acos_changes(ctx)
    if df is None or df.empty:
        return ok()
    if not ctx.ref_date or not col_date or not col_val:
        return ok()

    tmp = df.copy()
    tmp["_dt"] = pd.to_datetime(tmp[col_date], errors="coerce")
    tmp["_v"] = tmp[col_val].apply(as_float)
    tmp = tmp[tmp["_dt"].notna() & tmp["_v"].notna()].copy()
    if tmp.empty or len(tmp.index) < 2:
        return ok()

    # window: same as mapping for ACoS changes: last 90 days
    cutoff_90 = pd.Timestamp(ctx.ref_date - timedelta(days=90))
    tmp = tmp[tmp["_dt"] >= cutoff_90].copy()
    tmp = tmp.sort_values("_dt")

    vals = [_normalize_pct(v) for v in tmp["_v"].tolist()]
    if len(vals) < 2:
        return ok()

    for old, new in zip(vals[:-1], vals[1:]):
        if old == 0:
            return flag("Old iACoS value is 0; cannot compute relative magnitude.")
        mag = abs((new - old) / old) * 100.0
        if mag < 5.0 or mag > 25.0:
            return flag(f"Relative change magnitude {mag:.2f}% out of range (acceptable 5%–25%).")
    return ok()


# ---- Seller params (Row 7 only) ----
def eval_C004(ctx: DatabricksContext) -> ControlResult:
    sh, df, col, miss = seller_param_row7(ctx, "QuartileFactor", "QuartileFactor|quartile_factor|quartilefactor", no_data_flag=True)
    if miss:
        return miss
    v = as_float(df.iloc[0][col])
    return ok() if v is not None and abs(v - 1.0) < 1e-9 else flag(f"QuartileFactor={v} (expected 1.0).")


def eval_C005(ctx: DatabricksContext) -> ControlResult:
    sh, df, col, miss = seller_param_row7(ctx, "CurrentEpisolon", "CurrentEpisolon|CurrentEpsilon|currentepsilon|currentepisolon|current_epsilon", no_data_flag=True)
    if miss:
        return miss
    v = as_float(df.iloc[0][col])
    return ok() if v is not None and abs(v - 1.0) < 1e-9 else flag(f"CurrentEpisolon={v} (expected 1.0).")


# ---- Presence-based overrides/flags ----
def eval_C006(ctx: DatabricksContext) -> ControlResult:
    return any_rows(ctx, "PRODUCT_LEVEL_ACOS", "34_Product_Level_ACoS", no_data_ok=True)


def eval_C007(ctx: DatabricksContext) -> ControlResult:
    return any_rows(ctx, "CAMPAIGN_LEVEL_ACOS", "35_Campaign_Level_ACoS", no_data_ok=True)


def eval_C008(ctx: DatabricksContext) -> ControlResult:
    """
    Timeframe Boost:
      - OK if no data
      - OK if all rows are Expired
      - FLAG if any row is not Expired
    """
    sh, df = ds(ctx, "TIMEFRAME_BOOST", "27_Timeframe_Boost")
    if df is None or df.empty:
        return ok()

    status_col = find_col(df, ["status", "Status", "status_name", "statusname"])
    if not status_col:
        return flag(note_data_missing(sh or expected_tab_label("27_Timeframe_Boost"), "Status"))

    tmp = df.copy()
    tmp["_status"] = tmp[status_col].astype(str).fillna("").str.strip().str.lower()
    tmp = tmp[tmp["_status"] != ""]
    if tmp.empty:
        return ok()

    if (tmp["_status"] != "expired").any():
        n = int((tmp["_status"] != "expired").sum())
        return flag(f"Timeframe boost non-expired rows={n}.")
    return ok()


# ---- Removed ----
def eval_C009(ctx: DatabricksContext) -> ControlResult:
    return ok("REMOVED")


def eval_C010(ctx: DatabricksContext) -> ControlResult:
    return ok("REMOVED")


# ---- Negatives ----
_NEGATIVE_EXCEPTIONS = [
    "deal",
    "deals",
    "discount",
    "black friday",
    "cyber monday",
    "prime day",
    "holiday",
]


def _clean_cell_to_str(x) -> str:
    """
    Robust string cleaner for Excel-loaded cells:
      - preserves real text
      - converts NaN/None/"nan"/"none" to ""
      - trims whitespace
    """
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    if s.lower() in {"nan", "none"}:
        return ""
    return s


def _is_exception_negative(term: str) -> bool:
    """
    Exception is substring-based (as requested):
    if the negative contains any exception token anywhere -> exception.
    """
    t = _clean_cell_to_str(term).lower()
    if not t:
        return False
    return any(k in t for k in _NEGATIVE_EXCEPTIONS)


def eval_C011(ctx: DatabricksContext) -> ControlResult:
    """
    Account-level negatives:
      - Negative_Word nonblank
      - Product blank
      - OK if ALL are exceptions; FLAG if any non-exception exists
    No-data: OK
    """
    sh, df = ds(ctx, "NEGATIVES", "29_Negative_Keywords__Global")
    if df is None or df.empty:
        return ok()

    neg = find_col(df, ["negative_word", "negative word", "negative", "keyword"])
    prod = find_col(df, ["product", "asin", "targetasin"])
    if not neg:
        return ok()

    tmp = df.copy()

    # FIX: clean negatives safely (avoid "nan" strings)
    tmp["_neg"] = tmp[neg].apply(_clean_cell_to_str)
    tmp = tmp[tmp["_neg"] != ""]
    if tmp.empty:
        return ok()

    # FIX: clean product safely (avoid "nan" strings misclassifying as product-level)
    if prod:
        tmp["_prod"] = tmp[prod].apply(_clean_cell_to_str)
        acct = tmp[tmp["_prod"] == ""].copy()
    else:
        acct = tmp.copy()

    if acct.empty:
        return ok()

    non_exc = [x for x in acct["_neg"].tolist() if not _is_exception_negative(x)]
    if non_exc:
        return flag(f"Non-exception account-level negatives found: {len(non_exc)}.")
    return ok()


def eval_C012(ctx: DatabricksContext) -> ControlResult:
    """
    Product-level negatives:
      - Negative_Word nonblank
      - Product NOT blank
      - OK if ALL are exceptions; FLAG if any non-exception exists
    No-data: OK
    """
    sh, df = ds(ctx, "NEGATIVES", "29_Negative_Keywords__Global")
    if df is None or df.empty:
        return ok()

    neg = find_col(df, ["negative_word", "negative word", "negative", "keyword"])
    prod = find_col(df, ["product", "asin", "targetasin"])
    if not neg:
        return ok()

    tmp = df.copy()

    # FIX: clean negatives safely (avoid "nan" strings)
    tmp["_neg"] = tmp[neg].apply(_clean_cell_to_str)
    tmp = tmp[tmp["_neg"] != ""]
    if tmp.empty:
        return ok()

    if not prod:
        return ok()

    # FIX: clean product safely (avoid "nan" strings)
    tmp["_prod"] = tmp[prod].apply(_clean_cell_to_str)
    prod_df = tmp[tmp["_prod"] != ""].copy()
    if prod_df.empty:
        return ok()

    non_exc = [x for x in prod_df["_neg"].tolist() if not _is_exception_negative(x)]
    if non_exc:
        return flag(f"Non-exception product-level negatives found: {len(non_exc)}.")
    return ok()

# ---- Product Tag Completeness ----
def eval_C013(ctx: DatabricksContext) -> ControlResult:
    sh, df = ds(ctx, "CAMPAIGN_PERF", "14_Campaign_Performance_by_Adve")
    if df is None or df.empty:
        return flag(note_data_missing(expected_tab_label("14_Campaign_Performance_by_Adve"), "ASIN/Tags"))

    asin_col = find_col(df, ["asin"])
    if not asin_col:
        return flag(note_data_missing(sh or expected_tab_label("14_Campaign_Performance_by_Adve"), "ASIN"))

    tag_cols = []
    for t in ["tag1", "tag2", "tag3", "tag4", "tag5"]:
        c = find_col(df, [t])
        if c:
            tag_cols.append(c)
    if not tag_cols:
        return flag(note_data_missing(sh or expected_tab_label("14_Campaign_Performance_by_Adve"), "Tag1-Tag5"))

    active = [c for c in tag_cols if (df[c].astype(str).fillna("").str.strip() != "").any()]

    tag1_col = find_col(df, ["tag1"])
    if tag1_col:
        asin_has_values = (df[asin_col].astype(str).fillna("").str.strip() != "").any()
        tag1_has_values = (df[tag1_col].astype(str).fillna("").str.strip() != "").any()
        if asin_has_values and not tag1_has_values:
            return flag("Tag1 column present but no tags assigned to any ASIN.")

    if not active:
        return ok()

    elig = df[df[asin_col].astype(str).fillna("").str.strip() != ""].copy()
    if elig.empty:
        return ok()

    total = len(elig.index) * len(active)
    missing = 0
    for c in active:
        missing += (elig[c].astype(str).fillna("").str.strip() == "").sum()
    missing_pct = (missing / total) * 100.0 if total else 0.0

    if missing_pct >= 25.0:
        return flag(f"Missing tags {missing_pct:.2f}% (>=25%).")
    if missing_pct >= 10.0:
        return partial(f"Missing tags {missing_pct:.2f}% (>=10% and <25%).")
    return ok()


# ---- Branded/Competitor terms (fixed rows) ----
def eval_C014(ctx: DatabricksContext) -> ControlResult:
    sh, df = ds(ctx, "BRAND_COMP_TERMS", "30_Branded_and_Competitor_Terms")
    if df is None or df.empty or len(df.index) < 1:
        return flag(note_data_missing(expected_tab_label("30_Branded_and_Competitor_Terms"), "Row7"))
    total_terms = find_col(df, ["total_terms", "total terms", "terms", "count", "keyword_count", "total"])
    if not total_terms:
        return flag(note_data_missing(sh or expected_tab_label("30_Branded_and_Competitor_Terms"), "Total_Terms"))
    n = as_int(df.iloc[0][total_terms])
    if n is None:
        return flag(note_data_missing(sh or expected_tab_label("30_Branded_and_Competitor_Terms"), "Total_Terms"))
    return ok() if n >= 1 else flag(f"Total_Terms={n} (<1) on Row7.")


def eval_C015(ctx: DatabricksContext) -> ControlResult:
    sh, df = ds(ctx, "BRAND_COMP_TERMS", "30_Branded_and_Competitor_Terms")
    if df is None or df.empty or len(df.index) < 2:
        return flag(note_data_missing(expected_tab_label("30_Branded_and_Competitor_Terms"), "Row8"))
    total_terms = find_col(df, ["total_terms", "total terms", "terms", "count", "keyword_count", "total"])
    if not total_terms:
        return flag(note_data_missing(sh or expected_tab_label("30_Branded_and_Competitor_Terms"), "Total_Terms"))
    n = as_int(df.iloc[1][total_terms])
    if n is None:
        return flag(note_data_missing(sh or expected_tab_label("30_Branded_and_Competitor_Terms"), "Total_Terms (Row8)"))
    if n == 0:
        return flag("Total_Terms=0 on Row8.")
    if 1 <= n <= 2:
        return partial(f"Total_Terms={n} on Row8 (1–2).")
    return ok()


# ---- Unmanaged (end-date based) ----
def eval_C016(ctx: DatabricksContext) -> ControlResult:
    sh, df = ds(ctx, "UNMANAGED_ASIN", "26_Unmanaged_ASIN")
    if df is None or df.empty:
        return ok()
    if not getattr(ctx, "ref_date", None):
        return flag(note_data_missing(sh or expected_tab_label("26_Unmanaged_ASIN"), "Downloaded timestamp (ref_date)"))
    if df.shape[1] < 5:
        return flag(note_data_missing(sh or expected_tab_label("26_Unmanaged_ASIN"), "Unmanaged_End_Date"))
    end_col = df.columns[4]
    end_dates = pd.to_datetime(df[end_col], errors="coerce").dropna()
    if end_dates.empty:
        return flag(note_data_missing(sh or expected_tab_label("26_Unmanaged_ASIN"), "Unmanaged_End_Date"))
    if bool((end_dates.dt.date > ctx.ref_date).any()):
        n = int((end_dates.dt.date > ctx.ref_date).sum())
        return flag(f"Unmanaged ASIN end_date > ref_date rows={n}.")
    return ok()


def eval_C017(ctx: DatabricksContext) -> ControlResult:
    sh, df = ds(ctx, "UNMANAGED_BUDGET", "28_Unmanaged_Budget")
    if df is None or df.empty:
        return ok()
    if not getattr(ctx, "ref_date", None):
        return flag(note_data_missing(sh or expected_tab_label("28_Unmanaged_Budget"), "Downloaded timestamp (ref_date)"))
    if df.shape[1] < 7:
        return flag(note_data_missing(sh or expected_tab_label("28_Unmanaged_Budget"), "Unmanaged_End_Date"))
    end_col = df.columns[6]
    end_dates = pd.to_datetime(df[end_col], errors="coerce").dropna()
    if end_dates.empty:
        return flag(note_data_missing(sh or expected_tab_label("28_Unmanaged_Budget"), "Unmanaged_End_Date"))
    if bool((end_dates.dt.date > ctx.ref_date).any()):
        n = int((end_dates.dt.date > ctx.ref_date).sum())
        return flag(f"Unmanaged Budget end_date > ref_date rows={n}.")
    return ok()


def eval_C018(ctx: DatabricksContext) -> ControlResult:
    sh, df = ds(ctx, "UNMANAGED_CAMPAIGNS", "31_Unmanaged_campaigns")
    if df is None or df.empty:
        return ok()
    if not getattr(ctx, "ref_date", None):
        return flag(note_data_missing(sh or expected_tab_label("31_Unmanaged_campaigns"), "Downloaded timestamp (ref_date)"))
    if df.shape[1] < 12:
        return flag(note_data_missing(sh or expected_tab_label("31_Unmanaged_campaigns"), "Unmanaged_End_Date"))
    end_col = df.columns[11]
    end_dates = pd.to_datetime(df[end_col], errors="coerce").dropna()
    if end_dates.empty:
        return flag(note_data_missing(sh or expected_tab_label("31_Unmanaged_campaigns"), "Unmanaged_End_Date"))
    if bool((end_dates.dt.date > ctx.ref_date).any()):
        n = int((end_dates.dt.date > ctx.ref_date).sum())
        return flag(f"Unmanaged Campaigns end_date > ref_date rows={n}.")
    return ok()


def eval_C019(ctx: DatabricksContext) -> ControlResult:
    sh, df = ds(ctx, "UNMANAGED_CAMPAIGN_BUDGETS", "32_Unmanaged_Campaigns_Budget_O")
    if df is None or df.empty:
        return ok()
    if not getattr(ctx, "ref_date", None):
        return flag(note_data_missing(sh or expected_tab_label("32_Unmanaged_Campaigns_Budget_O"), "Downloaded timestamp (ref_date)"))
    if df.shape[1] < 7:
        return flag(note_data_missing(sh or expected_tab_label("32_Unmanaged_Campaigns_Budget_O"), "Unmanaged_End_Date"))
    end_col = df.columns[6]
    end_dates = pd.to_datetime(df[end_col], errors="coerce").dropna()
    if end_dates.empty:
        return flag(note_data_missing(sh or expected_tab_label("32_Unmanaged_Campaigns_Budget_O"), "Unmanaged_End_Date"))
    if bool((end_dates.dt.date > ctx.ref_date).any()):
        n = int((end_dates.dt.date > ctx.ref_date).sum())
        return flag(f"Unmanaged Campaign Budgets end_date > ref_date rows={n}.")
    return ok()


# ---- ARIS manual recommendations (presence-based) ----
def eval_C020(ctx: DatabricksContext) -> ControlResult:
    return any_rows(ctx, "ARIS_MANUAL_RECS", "41_ARIS__Manual_Recomendation", no_data_ok=True)


# ---- Portfolio controls ----
def eval_C021(ctx: DatabricksContext) -> ControlResult:
    sh, df = ds(ctx, "PORTFOLIO_INSIGHTS", "25_Portfolio_Insights_and_Confi")
    if df is None or df.empty:
        return ok()
    is_managed = find_col(df, ["ismanaged"])
    is_daily = find_col(df, ["isdailyvambaseline"])
    if not is_managed or not is_daily:
        return ok()
    elig = df[df[is_managed].apply(as_bool) == True]  # noqa: E712
    if elig.empty:
        return ok()
    flags = elig[is_daily].apply(as_bool).dropna()
    return flag("Managed portfolio(s) with IsDailyVamBaseline=True detected.") if (not flags.empty and flags.any()) else ok()


def eval_C022(ctx: DatabricksContext) -> ControlResult:
    sh, df = ds(ctx, "PORTFOLIO_INSIGHTS", "25_Portfolio_Insights_and_Confi")
    if df is None or df.empty:
        return ok()
    is_managed = find_col(df, ["ismanaged"])
    col = find_col(df, ["istargetacos"])
    if not is_managed or not col:
        return ok()
    elig = df[df[is_managed].apply(as_bool) == True]  # noqa: E712
    if elig.empty:
        return ok()
    vals = [as_bool(v) for v in elig[col].tolist() if as_bool(v) is not None]
    if not vals:
        return ok()
    if all(v is False for v in vals):
        return ok()
    if all(v is True for v in vals):
        return partial("All managed portfolios have IsTargetACoS=True.")
    return flag("Mixed IsTargetACoS values across managed portfolios (both True and False).")


def eval_C023(ctx: DatabricksContext) -> ControlResult:
    sh, df = ds(ctx, "PORTFOLIO_INSIGHTS", "25_Portfolio_Insights_and_Confi")
    if df is None or df.empty:
        return ok()
    is_managed = find_col(df, ["ismanaged"])
    col = find_col(df, ["isbudgetcap"])
    if not is_managed or not col:
        return ok()
    elig = df[df[is_managed].apply(as_bool) == True]  # noqa: E712
    if elig.empty:
        return ok()
    vals = [as_bool(v) for v in elig[col].tolist() if as_bool(v) is not None]
    if not vals:
        return ok()
    if all(v is False for v in vals):
        return ok()
    if all(v is True for v in vals):
        return partial("All managed portfolios have IsBudgetCap=True.")
    return flag("Mixed IsBudgetCap values across managed portfolios (both True and False).")


# ---- Seller Params (continued) ----
def eval_C024(ctx: DatabricksContext) -> ControlResult:
    sh, df, col, miss = seller_param_row7(ctx, "SelfService", "SelfService|selfservice", no_data_flag=True)
    if miss:
        return miss
    b = as_bool(df.iloc[0][col])
    if b is None:
        return flag(note_data_missing(sh or expected_tab_label("40_Seller_Parameter_Insights_Da"), "SelfService"))
    return flag("SelfService=True.") if b is True else ok()


def eval_C025(ctx: DatabricksContext) -> ControlResult:
    sh, df, col, miss = seller_param_row7(ctx, "MinBid", "MinBid|minbid|min_bid", no_data_flag=True)
    if miss:
        return miss
    v = as_float(df.iloc[0][col])
    if v is None:
        return flag(note_data_missing(sh or expected_tab_label("40_Seller_Parameter_Insights_Da"), "MinBid"))
    return ok() if (0.02 <= v <= 0.15) else flag(f"MinBid={v} (expected 0.02–0.15).")


def eval_C026(ctx: DatabricksContext) -> ControlResult:
    sh, df, col, miss = seller_param_row7(ctx, "MaxConversionRate", "MaxConversionRate|maxconversionrate|max_conversion_rate", no_data_flag=True)
    if miss:
        return miss
    v = as_float(df.iloc[0][col])
    if v is None:
        return flag(note_data_missing(sh or expected_tab_label("40_Seller_Parameter_Insights_Da"), "MaxConversionRate"))
    if abs(v - 25.00) <= 1e-6:
        return ok()
    if v > 25.00:
        return partial(f"MaxConversionRate={v} (>25).")
    return flag(f"MaxConversionRate={v} (<25).")


def eval_C027(ctx: DatabricksContext) -> ControlResult:
    sh, df, col1, miss1 = seller_param_row7(ctx, "PromoteKeywordMinClicks", "PromoteKeywordMinClicks|promotekeywordminclicks|promote_keyword_min_clicks", no_data_flag=True)
    if miss1:
        return miss1
    sh, df, col2, miss2 = seller_param_row7(ctx, "NegateKeywordMinClicks", "NegateKeywordMinClicks|negatekeywordminclicks|negate_keyword_min_clicks", no_data_flag=True)
    if miss2:
        return miss2
    v1 = as_float(df.iloc[0][col1])
    v2 = as_float(df.iloc[0][col2])
    if v1 is None or v2 is None:
        return flag(note_data_missing(sh or expected_tab_label("40_Seller_Parameter_Insights_Da"), "Promote/Negate"))
    if abs(v1) < 1e-9 and abs(v2) < 1e-9:
        return ok()
    return flag(f"PromoteKeywordMinClicks={v1}, NegateKeywordMinClicks={v2} (expected both 0).")


def eval_C028(ctx: DatabricksContext) -> ControlResult:
    sh, df, col, miss = seller_param_row7(ctx, "BudgetManagement", "BudgetManagement|budgetmanagement|budget_management", no_data_flag=True)
    if miss:
        return miss
    b = as_bool(df.iloc[0][col])
    if b is None:
        return flag(note_data_missing(sh or expected_tab_label("40_Seller_Parameter_Insights_Da"), "BudgetManagement"))
    return ok() if b is True else flag(f"BudgetManagement={b} (expected True).")


def eval_C029(ctx: DatabricksContext) -> ControlResult:
    sh, df, col, miss = seller_param_row7(ctx, "PlacementModifierManagement", "PlacementModifierManagement|placementmodifiermanagement|placement_modifier_management", no_data_flag=True)
    if miss:
        return miss
    b = as_bool(df.iloc[0][col])
    if b is None:
        return flag(note_data_missing(sh or expected_tab_label("40_Seller_Parameter_Insights_Da"), "PlacementModifierManagement"))
    return ok() if b is True else flag(f"PlacementModifierManagement={b} (expected True).")


def eval_C030(ctx: DatabricksContext) -> ControlResult:
    sh, df, col, miss = seller_param_row7(ctx, "MktStreamHourlyBidAdjustments", "MktStreamHourlyBidAdjustments|mktstreamhourlybidadjustments|mkt_stream_hourly_bid_adjustments", no_data_flag=True)
    if miss:
        return miss
    b = as_bool(df.iloc[0][col])
    if b is None:
        return flag(note_data_missing(sh or expected_tab_label("40_Seller_Parameter_Insights_Da"), "MktStreamHourlyBidAdjustments"))
    return ok() if b is True else flag(f"MktStreamHourlyBidAdjustments={b} (expected True).")


def eval_C031(ctx: DatabricksContext) -> ControlResult:
    sh, df, col, miss = seller_param_row7(ctx, "AutomaticallyImportCampaigns", "AutomaticallyImportCampaigns|automaticallyimportcampaigns|automatically_import_campaigns", no_data_flag=True)
    if miss:
        return miss
    b = as_bool(df.iloc[0][col])
    if b is None:
        return flag(note_data_missing(sh or expected_tab_label("40_Seller_Parameter_Insights_Da"), "AutomaticallyImportCampaigns"))
    return flag("AutomaticallyImportCampaigns=True.") if b is True else ok()


def eval_C032(ctx: DatabricksContext) -> ControlResult:
    sh, df, col, miss = seller_param_row7(ctx, "StopAudienceAutoLink", "StopAudienceAutoLink|stopaudienceautolink|stop_audience_auto_link", no_data_flag=True)
    if miss:
        return miss
    b = as_bool(df.iloc[0][col])
    if b is None:
        return flag(note_data_missing(sh or expected_tab_label("40_Seller_Parameter_Insights_Da"), "StopAudienceAutoLink"))
    return flag("StopAudienceAutoLink=True.") if b is True else ok()


def eval_C033(ctx: DatabricksContext) -> ControlResult:
    sh, df, col, miss = seller_param_row7(ctx, "IsB2bPlacementManagement", "IsB2bPlacementManagement|isb2bplacementmanagement|is_b2b_placement_management|b2bplacementmanagement", no_data_flag=True)
    if miss:
        return miss
    b = as_bool(df.iloc[0][col])
    if b is None:
        return flag(note_data_missing(sh or expected_tab_label("40_Seller_Parameter_Insights_Da"), "IsB2bPlacementManagement"))
    return ok() if b is True else flag(f"IsB2bPlacementManagement={b} (expected True).")


def eval_C034(ctx: DatabricksContext) -> ControlResult:
    sh, df, col, miss = seller_param_row7(ctx, "HasDisplayPromote", "HasDisplayPromote|hasdisplaypromote|has_display_promote", no_data_flag=True)
    if miss:
        return miss
    b = as_bool(df.iloc[0][col])
    if b is None:
        return flag(note_data_missing(sh or expected_tab_label("40_Seller_Parameter_Insights_Da"), "HasDisplayPromote"))
    return ok() if b is True else flag("HasDisplayPromote=False (expected True).")


def eval_C035(ctx: DatabricksContext) -> ControlResult:
    sh, df, col, miss = seller_param_row7(ctx, "ChangeSBV", "ChangeSBV|changesbv|change_sbv", no_data_flag=True)
    if miss:
        return miss
    b = as_bool(df.iloc[0][col])
    if b is None:
        return flag(note_data_missing(sh or expected_tab_label("40_Seller_Parameter_Insights_Da"), "ChangeSBV"))
    return ok() if b is True else flag(f"ChangeSBV={b} (expected True).")


# ---- RBO ----
def eval_C036(ctx: DatabricksContext) -> ControlResult:
    """
    Tab: 33_RBO_Configuration_Insights

    FLAG if any RBO rule exists:
      - Data_Type (Col A) == "Rules"
      - and Rule_Name (Col F) OR Setting_Value (Col D) is nonblank

    OK if tab empty OR no rules found.
    """
    sh, df = ds(ctx, "RBO_CONFIG", "33_RBO_Configuration_Insights")
    if df is None or df.empty:
        return ok()

    tab = sh or expected_tab_label("33_RBO_Configuration_Insights")

    # Require at least cols A..F
    if df.shape[1] < 6:
        return flag(note_data_missing(tab, "Data_Type/Setting_Value/Rule_Name"))

    data_type_col = df.columns[0]    # A
    setting_val_col = df.columns[3]  # D
    rule_name_col = df.columns[5]    # F

    tmp = df.copy()
    tmp["_dtype"] = tmp[data_type_col].astype(str).fillna("").str.strip().str.lower()
    tmp["_setting"] = tmp[setting_val_col].astype(str).fillna("").str.strip()
    tmp["_rulename"] = tmp[rule_name_col].astype(str).fillna("").str.strip()

    rules = tmp[(tmp["_dtype"] == "rules") & ((tmp["_rulename"] != "") | (tmp["_setting"] != ""))].copy()
    if rules.empty:
        return ok()

    n = int(len(rules.index))
    examples = rules["_rulename"].replace("", np.nan).dropna().head(3).tolist()
    ex_txt = ", ".join(examples) if examples else "Rule_Name not provided"
    return flag(f"{tab}: {n} RBO rule(s) found. Examples: {ex_txt}.")


# ---- SPT Coverage ----
def eval_C037(ctx: DatabricksContext) -> ControlResult:
    sh10, df10 = ds(ctx, "CAMPAIGNS_BY_TYPE", "10_Campaigns_Grouped_by_QT_Camp")
    sh18, df18 = ds(ctx, "PERF_BY_CATEGORY", "18_Performance_by_Category")
    if (df10 is None or df10.empty) or (df18 is None or df18.empty):
        return ok()

    asin_cnt = find_col(df18, ["asincount", "asin_count", "asin count"])
    if not asin_cnt:
        asin_cnt = get_col_by_letter(df18, "B")
    if not asin_cnt:
        return ok()

    req = int((df18[asin_cnt].apply(as_int).fillna(0) >= 30).sum())

    subtype = find_col(df10, ["campaignsubtype"])
    campaigns = find_col(df10, ["campaigns"])
    if not subtype or not campaigns:
        return ok()

    rows = df10[df10[subtype].astype(str).str.strip().str.upper() == "SPT"]
    spt = int(as_int(rows[campaigns].dropna().iloc[0]) or 0) if not rows.empty else 0

    if spt == 0:
        return flag("No SPT campaigns.")
    if req > 0 and spt < req:
        return partial(f"SPT campaigns ({spt}) < required categories ({req}).")
    return ok()


# ---- CatchAll vs WATM ----
def eval_C038(ctx: DatabricksContext) -> ControlResult:
    """
    CatchAll REQUIRED only if:
      - AvgCPC < 0.50 (03 KPI B10)
      - ASIN_count > 200 (count ASINs in 14 col D)
      - ACoS_target < 15% (41 U7 row7)
    If any condition missing => WATM alone OK.
    Detect WATM from QT subtype table (10) CampaignSubType=='WATM'.
    Detect CatchAll by CampaignName patterns in campaign report (08).
    """
    sh10, df10 = ds(ctx, "CAMPAIGNS_BY_TYPE", "10_Campaigns_Grouped_by_QT_Camp")
    sh08, df08 = ds(ctx, "CAMPAIGN_REPORT", "08_Campaign_Report")
    sh03, df03 = ds(ctx, "YEARLY_KPIS", "03_Yearly_KPIs")
    sh14, df14 = ds(ctx, "CAMPAIGN_PERF", "14_Campaign_Perf")
    sh41, df41 = ds(ctx, "SELLER_PARAMS", "40_Seller_Params")

    if df10 is None or df10.empty or df08 is None or df08.empty:
        return flag(note_data_missing(expected_tab_label("10_Campaigns_Grouped_by_QT_Camp"), "CampaignSubType/CampaignName"))

    subtype = find_col(df10, ["campaignsubtype"])
    campaigns = find_col(df10, ["campaigns"])
    watm = 0
    if subtype and campaigns:
        rows = df10[df10[subtype].astype(str).str.strip().str.upper() == "WATM"]
        watm = int(as_int(rows[campaigns].dropna().iloc[0]) or 0) if not rows.empty else 0

    cname = find_col(df08, ["campaignname", "campaign name", "name"])
    if not cname:
        cname = get_col_by_letter(df08, "B")
    catchall = False
    if cname:
        pat = re.compile(r"(catch[\s\-]?all|un_watm_)", re.IGNORECASE)
        catchall = df08[cname].astype(str).apply(lambda s: bool(pat.search(s))).any()

    required = True

    avg_cpc = None
    if df03 is not None and not df03.empty:
        col_b = get_col_by_letter(df03, "B")
        if col_b:
            col_a = get_col_by_letter(df03, "A")
            if col_a:
                rows = df03[df03[col_a].astype(str).str.contains("AvgCPC", case=False, na=False)]
                if not rows.empty:
                    avg_cpc = as_float(rows.iloc[0][col_b])
            if avg_cpc is None and len(df03.index) >= 10:
                avg_cpc = as_float(df03.iloc[9][col_b])
    if avg_cpc is None:
        required = False
    else:
        required = required and (avg_cpc < 0.50)

    asin_count = None
    if df14 is not None and not df14.empty:
        asin_col = find_col(df14, ["asin"])
        if not asin_col:
            asin_col = get_col_by_letter(df14, "D")
        if asin_col:
            asin_count = int((df14[asin_col].astype(str).fillna("").str.strip() != "").sum())
    if asin_count is None:
        required = False
    else:
        required = required and (asin_count > 200)

    acos_target = None
    if df41 is not None and not df41.empty:
        col = find_col(df41, ["acostarget", "acos_target", "targetacos", "iacos", "iacos_percent"])
        if col:
            acos_target = as_float(df41.iloc[0][col])
            if acos_target is not None:
                acos_target = _normalize_pct(acos_target)
        else:
            col_u = get_col_by_letter(df41, "U")
            if col_u:
                acos_target = as_float(df41.iloc[0][col_u])
                if acos_target is not None:
                    acos_target = _normalize_pct(acos_target)
    if acos_target is None:
        required = False
    else:
        required = required and (acos_target < 15.0)

    if not required:
        if watm > 0 or catchall:
            if catchall and watm == 0:
                return partial("CatchAll present but not required; WATM missing.")
            return ok()
        return flag("Neither WATM nor CatchAll detected.")

    if catchall:
        return ok()
    if watm > 0 and not catchall:
        return partial("CatchAll required but only WATM detected.")
    return flag("CatchAll required but missing (and no WATM).")


# ---- ATM Coverage for Top Sellers ----
def eval_C039(ctx: DatabricksContext) -> ControlResult:
    sh, df = ds(ctx, "CAMPAIGN_PERF", "14_Campaign_Performance_by_Adve")
    if df is None or df.empty:
        return flag(note_data_missing(expected_tab_label("14_Campaign_Performance_by_Adve"), "ASIN/Orders/ATM_Spend"))

    asin = find_col(df, ["asin"])
    orders = find_col(df, ["orders", "purchases"])
    tier = find_col(df, ["tier"])
    atm_spend = find_col(df, ["atm_spend", "atm spend", "atmspend", "atm"])

    # This control is ATM-based -> require ATM_Spend
    if not asin or not orders or not atm_spend:
        missing = []
        if not asin:
            missing.append("ASIN")
        if not orders:
            missing.append("Orders")
        if not atm_spend:
            missing.append("ATM_Spend")
        return flag(note_data_missing(sh or expected_tab_label("14_Campaign_Performance_by_Adve"), "/".join(missing)))

    days = getattr(ctx, "window_days", None) or 30

    tmp = df.copy()
    tmp["_asin"] = tmp[asin].astype(str).fillna("").str.strip()
    tmp = tmp[tmp["_asin"] != ""].copy()
    if tmp.empty:
        return ok()

    tmp["_orders"] = tmp[orders].apply(as_float).fillna(0.0)
    tmp["_orders_per_day"] = tmp["_orders"] / float(days)
    tmp["_atm"] = tmp[atm_spend].apply(as_float).fillna(0.0)

    # PARTIAL: low-velocity ASINs still receiving ATM spend
    cond_partial = (tmp["_orders_per_day"] < 2.0) & (tmp["_atm"] > 0.0)
    if cond_partial.any():
        n = int(cond_partial.sum())
        return partial(f"{n} ASIN(s) with <2 orders/day have ATM spend > 0.")

    # FLAG: Tier 30 + high velocity but no ATM spend (only if Tier exists)
    if tier:
        tmp["_tier"] = tmp[tier].apply(as_int)
        cond_flag = (tmp["_tier"] == 30) & (tmp["_orders_per_day"] > 2.0) & (tmp["_atm"] == 0.0)
        if cond_flag.any():
            n = int(cond_flag.sum())
            return flag(f"{n} Tier 30 ASIN(s) with >2 orders/day have ATM spend = 0.")

    return ok()

# ---- BA → BAK Translation ----
def eval_C040(ctx: DatabricksContext) -> ControlResult:
    sh, df = ds(ctx, "CAMPAIGNS_BY_TYPE", "10_Campaigns_Grouped_by_QT_Camp")
    if df is None or df.empty:
        return flag(note_data_missing(expected_tab_label("10_Campaigns_Grouped_by_QT_Camp"), "CampaignSubType/Campaigns"))
    subtype = find_col(df, ["campaignsubtype"])
    campaigns = find_col(df, ["campaigns"])
    if not subtype or not campaigns:
        return flag(note_data_missing(sh or expected_tab_label("10_Campaigns_Grouped_by_QT_Camp"), "CampaignSubType/Campaigns"))

    def count(label: str) -> int:
        rows = df[df[subtype].astype(str).str.strip().str.upper() == label.upper()]
        if rows.empty:
            return 0
        v = as_int(rows[campaigns].dropna().iloc[0]) if not rows[campaigns].dropna().empty else 0
        return int(v or 0)

    ba = count("BA")
    bak = count("BAK")

    if ba > 0 and bak == 0:
        return flag(f"BA campaigns={ba} but BAK campaigns=0.")
    if ba > 0 and bak > 0 and ba > bak:
        return partial(f"BA campaigns={ba} > BAK campaigns={bak}.")
    return ok()


# ---- Branded vs Non-Branded Mix ----
def eval_C041(ctx: DatabricksContext) -> ControlResult:
    sh, df = ds(ctx, "SEARCH_TERMS_BY_CATEGORY", "12_Search_Terms_by_Category")
    if df is None or df.empty:
        return flag(note_data_missing(expected_tab_label("12_Search_Terms_by_Category"), "KeywordCategory/Spend_Pct"))

    cat = find_col(df, ["keywordcategory", "category"])
    spend_pct = find_col(df, ["spend_pct", "spend pct", "pct_spend", "perc_spend"])
    if not cat or not spend_pct:
        return flag(note_data_missing(sh or expected_tab_label("12_Search_Terms_by_Category"), "KeywordCategory/Spend_Pct"))

    # Branded row (case-insensitive exact match)
    rows = df[df[cat].astype(str).str.strip().str.lower() == "branded"]
    if rows.empty:
        return flag("Branded row missing.")

    v = as_float(rows[spend_pct].dropna().iloc[0]) if not rows[spend_pct].dropna().empty else None
    if v is None:
        return flag("Branded spend pct missing.")

    v = _normalize_pct(v)

    # NEW THRESHOLDS:
    # FLAG: <1% OR >40%
    # PARTIAL: 1–<5% OR 25–40% (inclusive on 40, exclusive on 25? user said 25% till 40%)
    # OK: 5–25% (inclusive on 25)
    if v < 1.0:
        return flag(f"Branded spend {v:.2f}% (<1%).")
    if v > 40.0:
        return flag(f"Branded spend {v:.2f}% (>40%).")

    if 1.0 <= v < 5.0:
        return partial(f"Branded spend {v:.2f}% (1–<5%).")
    if 25.0 < v <= 40.0:
        return partial(f"Branded spend {v:.2f}% (>25–40%).")

    return ok()

# ---- SB Spend Share ----
def eval_C042(ctx: DatabricksContext) -> ControlResult:
    sh, df = ds(ctx, "CAMPAIGNS_BY_AMAZON", "09_Campaigns_Grouped_by_Amazon_")
    if df is None or df.empty:
        return flag(note_data_missing(expected_tab_label("09_Campaigns_Grouped_by_Amazon_"), "Campaign_Type/Perc_Spend"))
    ctype = find_col(df, ["campaign_type", "type"])
    pct = find_col(df, ["perc_spend", "pct_spend", "spend_pct", "spend pct"])
    if not ctype or not pct:
        return flag(note_data_missing(sh or expected_tab_label("09_Campaigns_Grouped_by_Amazon_"), "Campaign_Type/Perc_Spend"))
    rows = df[df[ctype].astype(str).str.strip().str.lower() == "sponsored brands"]
    if rows.empty:
        return flag("Sponsored Brands row missing.")
    v = as_float(rows[pct].dropna().iloc[0]) if not rows[pct].dropna().empty else None
    if v is None:
        return flag("Sponsored Brands spend pct missing.")
    v = _normalize_pct(v)
    return flag(f"SB spend {v:.2f}% (<1% or >25%).") if (v < 1.0 or v > 25.0) else ok()


# ---- Removed ----
def eval_C043(ctx: DatabricksContext) -> ControlResult:
    return ok("REMOVED")


def eval_C044(ctx: DatabricksContext) -> ControlResult:
    return ok("REMOVED")


# ---- SBV / SBTV spend share (optional rows) ----
def _eval_optional_spend_share_cap(ctx: DatabricksContext, row_label: str, cap: float) -> ControlResult:
    sh, df = ds(ctx, "CAMPAIGNS_BY_AMAZON", "09_Campaigns_Grouped_by_Amazon_")
    if df is None or df.empty:
        return ok()
    ctype = find_col(df, ["campaign_type", "type"])
    pct = find_col(df, ["perc_spend", "pct_spend", "spend_pct", "spend pct"])
    if not ctype or not pct:
        return ok()
    rows = df[df[ctype].astype(str).str.strip().str.lower() == row_label.lower()]
    if rows.empty:
        return ok()
    v = as_float(rows[pct].dropna().iloc[0]) if not rows[pct].dropna().empty else None
    if v is None:
        return ok()
    v = _normalize_pct(v)
    return flag(f"{row_label} spend {v:.2f}% (> {cap:.0f}%).") if (v < 0.0 or v > cap) else ok()


def eval_C045(ctx: DatabricksContext) -> ControlResult:
    return _eval_optional_spend_share_cap(ctx, "Sponsored Brand Video", cap=20.0)


def eval_C046(ctx: DatabricksContext) -> ControlResult:
    return _eval_optional_spend_share_cap(ctx, "Sponsored Brand TV Video", cap=10.0)


# ---- SD Defensive Coverage ----
def eval_C047(ctx: DatabricksContext) -> ControlResult:
    sh10, df10 = ds(ctx, "CAMPAIGNS_BY_TYPE", "10_Campaigns_Grouped_by_QT_Camp")
    sh18, df18 = ds(ctx, "PERF_BY_CATEGORY", "18_Performance_by_Category")
    if (df10 is None or df10.empty) or (df18 is None or df18.empty):
        return ok()

    asin_cnt = find_col(df18, ["asincount", "asin_count", "asin count"]) or get_col_by_letter(df18, "B")
    if not asin_cnt:
        return ok()
    req = int((df18[asin_cnt].apply(as_int).fillna(0) >= 30).sum())

    subtype = find_col(df10, ["campaignsubtype"])
    campaigns = find_col(df10, ["campaigns"])
    if not subtype or not campaigns:
        return ok()
    rows = df10[df10[subtype].astype(str).str.strip().str.upper() == "SD_SPT"]
    sd_spt = int(as_int(rows[campaigns].dropna().iloc[0]) or 0) if not rows.empty else 0

    if sd_spt == 0:
        return flag("No SD_SPT campaigns.")
    if req > 0 and sd_spt < req:
        return partial(f"SD_SPT campaigns ({sd_spt}) < required categories ({req}).")
    return ok()


# ---- SD VCPM spend share ----
def eval_C048(ctx: DatabricksContext) -> ControlResult:
    sh, df = ds(ctx, "SEARCH_TERMS_BY_CATEGORY", "12_Search_Terms_by_Category")
    if df is None or df.empty:
        return ok()
    cat = find_col(df, ["keywordcategory", "category"])
    spend_pct = find_col(df, ["spend_pct", "spend pct", "pct_spend", "perc_spend"])
    if not cat or not spend_pct:
        return ok()
    rows = df[df[cat].astype(str).str.strip().str.lower().str.contains("vcpm", na=False)]
    if rows.empty:
        return ok()
    v = as_float(rows[spend_pct].dropna().iloc[0]) if not rows[spend_pct].dropna().empty else None
    if v is None:
        return ok()
    v = _normalize_pct(v)
    return flag(f"VCPM spend share {v:.2f}% (>10%).") if v > 10.0 else ok()


# ---- SD VCPM sales share ----
def eval_C049(ctx: DatabricksContext) -> ControlResult:
    sh, df = ds(ctx, "SEARCH_TERMS_BY_CATEGORY", "12_Search_Terms_by_Category")
    if df is None or df.empty:
        return ok()
    cat = find_col(df, ["keywordcategory", "category"])
    sales = find_col(df, ["ad_sales", "adsales", "sales"])
    vcpm_sales = find_col(df, ["vcpm_sales", "vcpm sales"])
    if not cat or not sales:
        return ok()

    if vcpm_sales:
        rows = df[df[cat].astype(str).str.strip().str.lower().str.contains("vcpm", na=False)]
        if rows.empty:
            return ok()
        v = as_float(rows[vcpm_sales].dropna().iloc[0]) if not rows[vcpm_sales].dropna().empty else None
        total = as_float(df[sales].dropna().sum())
    else:
        rows = df[df[cat].astype(str).str.strip().str.lower().str.contains("vcpm", na=False)]
        if rows.empty:
            return ok()
        v = as_float(rows[sales].dropna().iloc[0]) if not rows[sales].dropna().empty else None
        total = as_float(df[sales].dropna().sum())

    if v is None or not total or total == 0:
        return ok()

    share = (v / total) * 100.0
    return flag(f"VCPM sales share {share:.2f}% (>20%).") if share > 20.0 else ok()


# ---- Budget X Amount of Campaigns ----
def eval_C050(ctx: DatabricksContext) -> ControlResult:
    sh, df = ds(ctx, "CAMPAIGNS_BY_TYPE", "10_Campaigns_Grouped_by_QT_Camp")
    if df is None or df.empty:
        return flag(note_data_missing(expected_tab_label("10_Campaigns_Grouped_by_QT_Camp"), "Campaigns"))

    campaigns_col = find_col(df, ["campaigns"]) or get_col_by_letter(df, "B")
    if not campaigns_col:
        return flag(note_data_missing(sh or expected_tab_label("10_Campaigns_Grouped_by_QT_Camp"), "Campaigns"))

    total_campaigns = df[campaigns_col].apply(as_int).fillna(0).sum()
    if total_campaigns <= 0:
        return flag("No campaigns found (cannot compute).")

    daily_budget = getattr(ctx, "daily_budget", None)
    if daily_budget is None:
        return flag("DATA MISSING: INPUT.DailyBudget")

    budget_per_campaign = float(daily_budget) / float(total_campaigns)

    if budget_per_campaign < 0.25 or budget_per_campaign > 10.0:
        return flag(f"Budget per campaign {budget_per_campaign:.2f} (<0.25 or >10).")
    return ok()


# ---- Out of Budget (Consecutive Days) ----
def eval_C051(ctx: DatabricksContext) -> ControlResult:
    sh, df = ds(ctx, "ACCOUNT_OOB", "36_Account_Out_of_Budget")
    if df is None or df.empty:
        return ok()
    dt = find_col(df, ["date", "day"]) or get_col_by_letter(df, "A")
    if not dt:
        return ok()

    dts = pd.to_datetime(df[dt], errors="coerce").dropna().dt.normalize()
    if dts.empty:
        return ok()

    uniq = sorted(set(dts.tolist()))
    if len(uniq) >= 3:
        return flag(f"{len(uniq)} distinct OOB days (>=3).")

    if len(uniq) >= 2:
        for a, b in zip(uniq[:-1], uniq[1:]):
            if (b - a).days == 1:
                return partial("2 consecutive OOB days.")
    return ok()


# ---- Manual / visibility ----
def eval_C052(ctx: DatabricksContext) -> ControlResult:
    return ok("MANUAL CHECK REQUIRED")


def eval_C053(ctx: DatabricksContext) -> ControlResult:
    return ok("MANUAL CHECK REQUIRED")


def eval_C054(ctx: DatabricksContext) -> ControlResult:
    return ok("MANUAL CHECK REQUIRED")


# =========================================================
# Registry + runner
# =========================================================
ALL_CONTROL_IDS: List[str] = [
    "C001",
    "C002",
    "C003",
    "C004",
    "C005",
    "C006",
    "C007",
    "C008",
    "C009",
    "C010",
    "C011",
    "C012",
    "C013",
    "C014",
    "C015",
    "C016",
    "C017",
    "C018",
    "C019",
    "C020",
    "C021",
    "C022",
    "C023",
    "C024",
    "C025",
    "C026",
    "C027",
    "C028",
    "C029",
    "C030",
    "C031",
    "C032",
    "C033",
    "C034",
    "C035",
    "C036",
    "C037",
    "C038",
    "C039",
    "C040",
    "C041",
    "C042",
    "C043",
    "C044",
    "C045",
    "C046",
    "C047",
    "C048",
    "C049",
    "C050",
    "C051",
    "C052",
    "C053",
    "C054",
]


CONTROL_REGISTRY: Dict[str, Callable[[DatabricksContext], ControlResult]] = {
    "C001": eval_C001,
    "C002": eval_C002,
    "C003": eval_C003,
    "C004": eval_C004,
    "C005": eval_C005,
    "C006": eval_C006,
    "C007": eval_C007,
    "C008": eval_C008,
    "C009": eval_C009,
    "C010": eval_C010,
    "C011": eval_C011,
    "C012": eval_C012,
    "C013": eval_C013,
    "C014": eval_C014,
    "C015": eval_C015,
    "C016": eval_C016,
    "C017": eval_C017,
    "C018": eval_C018,
    "C019": eval_C019,
    "C020": eval_C020,
    "C021": eval_C021,
    "C022": eval_C022,
    "C023": eval_C023,
    "C024": eval_C024,
    "C025": eval_C025,
    "C026": eval_C026,
    "C027": eval_C027,
    "C028": eval_C028,
    "C029": eval_C029,
    "C030": eval_C030,
    "C031": eval_C031,
    "C032": eval_C032,
    "C033": eval_C033,
    "C034": eval_C034,
    "C035": eval_C035,
    "C036": eval_C036,
    "C037": eval_C037,
    "C038": eval_C038,
    "C039": eval_C039,
    "C040": eval_C040,
    "C041": eval_C041,
    "C042": eval_C042,
    "C043": eval_C043,
    "C044": eval_C044,
    "C045": eval_C045,
    "C046": eval_C046,
    "C047": eval_C047,
    "C048": eval_C048,
    "C049": eval_C049,
    "C050": eval_C050,
    "C051": eval_C051,
    "C052": eval_C052,
    "C053": eval_C053,
    "C054": eval_C054,
}


def evaluate_all(ctx: DatabricksContext) -> Dict[str, ControlResult]:
    """
    Deterministic: iterate SSOT order (Mapping v8) and run the registered function per control.
    """
    results: Dict[str, ControlResult] = {}
    for cid in ALL_CONTROL_IDS:
        fn = CONTROL_REGISTRY.get(cid)
        if not fn:
            results[cid] = ok()
            continue
        try:
            results[cid] = fn(ctx)
        except Exception as e:
            # never crash the agent; surface as FLAG with exception note
            results[cid] = ControlResult(cfg.STATUS_FLAG, f"EXCEPTION in {cid}: {type(e).__name__}: {e}")
    return results
