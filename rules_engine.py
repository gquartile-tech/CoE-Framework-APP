# rules_engine.py
from __future__ import annotations

import re
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
    note = (note or "").strip()
    if not note:
        note = "FLAG triggered but no deterministic note provided."
    return ControlResult(cfg.STATUS_FLAG, note)


def partial(note: str = "") -> ControlResult:
    return ControlResult(cfg.STATUS_PARTIAL, note or "")


def expected_tab_label(dataset_key: str) -> str:
    prefixes = getattr(cfg, "TAB_CANDIDATES", {}).get(dataset_key, [])
    return prefixes[0] if prefixes else dataset_key


def obs(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    return text if text.startswith("Observed:") else f"Observed: {text}"


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
    for cand in candidates:
        cn = norm(cand)
        if cn in cols_norm:
            return cols_norm[cn]
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
    cache = getattr(ctx, "_dataset_cache", None)
    if cache is None:
        cache = {}
        setattr(ctx, "_dataset_cache", cache)

    ck1 = ("ds", str(dataset_key))
    if ck1 in cache:
        sh, df = cache[ck1]
    else:
        sh, df = get_dataset(ctx, dataset_key)
        cache[ck1] = (sh, df)

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
    return flag(obs(f"{n} active row(s) were found in {tab}, indicating that this condition is currently present."))


def _normalize_pct(v: float) -> float:
    return v * 100.0 if 0 < v <= 1.0 else float(v)


# =========================================================
# Portfolio threshold helper
# =========================================================
def _portfolio_threshold_check(
    ctx: DatabricksContext,
    col_candidates: List[str],
    condition_label: str,
) -> ControlResult:
    """
    Shared logic for C019/C020/C021.
    Counts managed portfolios where the target column is True.
    >50% → FLAG, 25-50% → PARTIAL, <25% → OK.
    No data or no managed portfolios → OK.
    """
    sh, df = ds(ctx, "PORTFOLIO_INSIGHTS", "25_Portfolio_Insights_and_Confi")
    if df is None or df.empty:
        return ok()

    is_managed = find_col(df, ["ismanaged"])
    if not is_managed:
        return ok()

    elig = df[df[is_managed].apply(as_bool) == True].copy()  # noqa: E712
    total_managed = len(elig)
    if total_managed == 0:
        return ok()

    col = find_col(elig, col_candidates)
    if not col:
        return ok()

    active_count = int((elig[col].apply(as_bool) == True).sum())  # noqa: E712
    pct = (active_count / total_managed) * 100.0

    if pct > 50.0:
        return flag(obs(
            f"{active_count} of {total_managed} managed portfolios ({pct:.1f}%) have {condition_label} enabled, "
            f"above the 50% threshold."
        ))
    if pct >= 25.0:
        return partial(obs(
            f"{active_count} of {total_managed} managed portfolios ({pct:.1f}%) have {condition_label} enabled, "
            f"between the 25%–50% caution range."
        ))
    return ok()


# =========================================================
# Controls C001–C048
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
    PARTIAL: any gap <14 days in last 180 days (but not in 90d)
    OK: otherwise
    """
    sh, df, col_date, _ = _load_acos_changes(ctx)
    if df is None or df.empty:
        return ok()
    if not ctx.ref_date or not col_date:
        return ok()

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
        return flag(obs(f"ACoS changes were made too frequently in the last 90 days. Minimum gap was {g90} days, below the 14-day threshold."))
    g180 = min_gap_days(180)
    if g180 is not None and g180 < 14:
        return partial(obs(f"ACoS changes were relatively frequent in the last 180 days. Minimum gap was {g180} days, below the 14-day threshold."))
    return ok()


def eval_C002(ctx: DatabricksContext) -> ControlResult:
    """FLAG: change count in last 90 days > 5"""
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
    return flag(obs(f"ACoS target was changed {changes_90} times in the last 90 days, above the limit of 5 changes.")) if changes_90 > 5 else ok()


def eval_C003(ctx: DatabricksContext) -> ControlResult:
    """FLAG: any relative magnitude <5% or >25% between consecutive changes in last 90 days"""
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

    cutoff_90 = pd.Timestamp(ctx.ref_date - timedelta(days=90))
    tmp = tmp[tmp["_dt"] >= cutoff_90].copy()
    tmp = tmp.sort_values("_dt")

    vals = [_normalize_pct(v) for v in tmp["_v"].tolist()]
    if len(vals) < 2:
        return ok()

    for old, new in zip(vals[:-1], vals[1:]):
        if old == 0:
            return flag(obs("The previous iACoS value was 0, so relative change magnitude could not be calculated."))
        mag = abs((new - old) / old) * 100.0
        if mag < 5.0 or mag > 25.0:
            return flag(obs(f"ACoS change magnitude was outside the acceptable range. Relative change was {mag:.2f}%, versus the expected 5%–25% band."))
    return ok()


# ---- C004/C005: Seller params — QuartileFactor / CurrentEpisolon ----
def eval_C004(ctx: DatabricksContext) -> ControlResult:
    sh, df, col, miss = seller_param_row7(ctx, "QuartileFactor", "QuartileFactor|quartile_factor|quartilefactor", no_data_flag=True)
    if miss:
        return miss
    v = as_float(df.iloc[0][col])
    return ok() if v is not None and abs(v - 1.0) < 1e-9 else flag(obs(f"QuartileFactor is not aligned with the expected setup. Current value is {v}; expected 1.0."))


def eval_C005(ctx: DatabricksContext) -> ControlResult:
    sh, df, col, miss = seller_param_row7(ctx, "CurrentEpisolon", "CurrentEpisolon|CurrentEpsilon|currentepsilon|currentepisolon|current_epsilon", no_data_flag=True)
    if miss:
        return miss
    v = as_float(df.iloc[0][col])
    return ok() if v is not None and abs(v - 1.0) < 1e-9 else flag(obs(f"CurrentEpisolon is not aligned with the expected setup. Current value is {v}; expected 1.0."))


# ---- C006/C007: Presence-based ACoS overrides ----
def eval_C006(ctx: DatabricksContext) -> ControlResult:
    return any_rows(ctx, "PRODUCT_LEVEL_ACOS", "34_Product_Level_ACoS", no_data_ok=True)


def eval_C007(ctx: DatabricksContext) -> ControlResult:
    return any_rows(ctx, "CAMPAIGN_LEVEL_ACOS", "35_Campaign_Level_ACoS", no_data_ok=True)


# ---- C008: Timeframe Boost ----
def eval_C008(ctx: DatabricksContext) -> ControlResult:
    """OK if no data or all rows Expired. FLAG if any row is not Expired."""
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
        return flag(obs(f"{n} timeframe boost row(s) are still active and not marked as expired."))
    return ok()


# ---- C009/C010: Negative Keywords ----
_NEGATIVE_EXCEPTIONS = [
    "deal", "deals", "discount", "black friday",
    "cyber monday", "prime day", "holiday",
]


def _clean_cell_to_str(x) -> str:
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
    t = _clean_cell_to_str(term).lower()
    if not t:
        return False
    return any(k in t for k in _NEGATIVE_EXCEPTIONS)


def eval_C009(ctx: DatabricksContext) -> ControlResult:
    """Account-level negatives: Negative_Word nonblank, Product blank. FLAG if any non-exception."""
    sh, df = ds(ctx, "NEGATIVES", "29_Negative_Keywords__Global")
    if df is None or df.empty:
        return ok()

    neg = find_col(df, ["negative_word", "negative word", "negative", "keyword"])
    prod = find_col(df, ["product", "asin", "targetasin"])
    if not neg:
        return ok()

    tmp = df.copy()
    tmp["_neg"] = tmp[neg].apply(_clean_cell_to_str)
    tmp = tmp[tmp["_neg"] != ""]
    if tmp.empty:
        return ok()

    if prod:
        tmp["_prod"] = tmp[prod].apply(_clean_cell_to_str)
        acct = tmp[tmp["_prod"] == ""].copy()
    else:
        acct = tmp.copy()

    if acct.empty:
        return ok()

    non_exc = [x for x in acct["_neg"].tolist() if not _is_exception_negative(x)]
    if non_exc:
        return flag(obs(f"{len(non_exc)} non-exception account-level negative keyword(s) were found."))
    return ok()


def eval_C010(ctx: DatabricksContext) -> ControlResult:
    """Product-level negatives: Negative_Word nonblank, Product NOT blank. FLAG if any non-exception."""
    sh, df = ds(ctx, "NEGATIVES", "29_Negative_Keywords__Global")
    if df is None or df.empty:
        return ok()

    neg = find_col(df, ["negative_word", "negative word", "negative", "keyword"])
    prod = find_col(df, ["product", "asin", "targetasin"])
    if not neg:
        return ok()

    tmp = df.copy()
    tmp["_neg"] = tmp[neg].apply(_clean_cell_to_str)
    tmp = tmp[tmp["_neg"] != ""]
    if tmp.empty:
        return ok()

    if not prod:
        return ok()

    tmp["_prod"] = tmp[prod].apply(_clean_cell_to_str)
    prod_df = tmp[tmp["_prod"] != ""].copy()
    if prod_df.empty:
        return ok()

    non_exc = [x for x in prod_df["_neg"].tolist() if not _is_exception_negative(x)]
    if non_exc:
        return flag(obs(f"{len(non_exc)} non-exception product-level negative keyword(s) were found."))
    return ok()


# ---- C011: Product Tag Completeness ----
def eval_C011(ctx: DatabricksContext) -> ControlResult:
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
            return flag(obs("Tag1 column is present, but no ASINs have Tag1 assigned."))

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
        return flag(obs(f"Tag completeness is low. {missing_pct:.2f}% of required tag fields are missing, above the 25% threshold."))
    if missing_pct >= 10.0:
        return partial(obs(f"Tag completeness is below target. {missing_pct:.2f}% of required tag fields are missing."))
    return ok()


# ---- C012/C013: Branded/Competitor terms ----
def eval_C012(ctx: DatabricksContext) -> ControlResult:
    sh, df = ds(ctx, "BRAND_COMP_TERMS", "30_Branded_and_Competitor_Terms")
    if df is None or df.empty or len(df.index) < 1:
        return flag(note_data_missing(expected_tab_label("30_Branded_and_Competitor_Terms"), "Row7"))
    total_terms = find_col(df, ["total_terms", "total terms", "terms", "count", "keyword_count", "total"])
    if not total_terms:
        return flag(note_data_missing(sh or expected_tab_label("30_Branded_and_Competitor_Terms"), "Total_Terms"))
    n = as_int(df.iloc[0][total_terms])
    if n is None:
        return flag(note_data_missing(sh or expected_tab_label("30_Branded_and_Competitor_Terms"), "Total_Terms"))
    return ok() if n >= 1 else flag(obs(f"Total branded/competitor terms on Row 7 is {n}, below the minimum of 1."))


def eval_C013(ctx: DatabricksContext) -> ControlResult:
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
        return flag(obs("Total branded/competitor terms on Row 8 is 0."))
    if 1 <= n <= 2:
        return partial(obs(f"Total branded/competitor terms on Row 8 is {n}, which is below the preferred level."))
    return ok()


# ---- C014–C017: Unmanaged (end-date based) ----
def eval_C014(ctx: DatabricksContext) -> ControlResult:
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
        return flag(obs(f"{n} unmanaged ASIN row(s) have an end date after the reference date, so they are still active."))
    return ok()


def eval_C015(ctx: DatabricksContext) -> ControlResult:
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
        return flag(obs(f"{n} unmanaged budget row(s) have an end date after the reference date, so they are still active."))
    return ok()


def eval_C016(ctx: DatabricksContext) -> ControlResult:
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
        return flag(obs(f"{n} unmanaged campaign row(s) have an end date after the reference date, so they are still active."))
    return ok()


def eval_C017(ctx: DatabricksContext) -> ControlResult:
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
        return flag(obs(f"{n} unmanaged campaign budget row(s) have an end date after the reference date, so they are still active."))
    return ok()


# ---- C018: ARIS Manual Recommendations ----
def eval_C018(ctx: DatabricksContext) -> ControlResult:
    return any_rows(ctx, "ARIS_MANUAL_RECS", "41_ARIS__Manual_Recomendation", no_data_ok=True)


# ---- C019/C020/C021: Portfolio threshold controls ----
def eval_C019(ctx: DatabricksContext) -> ControlResult:
    """IsDailyVamBaseline: >50% managed portfolios → FLAG, 25-50% → PARTIAL, <25% → OK"""
    return _portfolio_threshold_check(
        ctx,
        col_candidates=["isdailyvambaseline", "IsDailyVamBaseline"],
        condition_label="IsDailyVamBaseline",
    )


def eval_C020(ctx: DatabricksContext) -> ControlResult:
    """IsTargetACoS: >50% managed portfolios → FLAG, 25-50% → PARTIAL, <25% → OK"""
    return _portfolio_threshold_check(
        ctx,
        col_candidates=["istargetacos", "IsTargetAcos", "IsTargetACoS"],
        condition_label="IsTargetACoS",
    )


def eval_C021(ctx: DatabricksContext) -> ControlResult:
    """IsBudgetCap: >50% managed portfolios → FLAG, 25-50% → PARTIAL, <25% → OK"""
    return _portfolio_threshold_check(
        ctx,
        col_candidates=["isbudgetcap", "IsBudgetCap"],
        condition_label="IsBudgetCap",
    )


# ---- C022–C033: Seller Params ----
def eval_C022(ctx: DatabricksContext) -> ControlResult:
    sh, df, col, miss = seller_param_row7(ctx, "SelfService", "SelfService|selfservice", no_data_flag=True)
    if miss:
        return miss
    b = as_bool(df.iloc[0][col])
    if b is None:
        return flag(note_data_missing(sh or expected_tab_label("40_Seller_Parameter_Insights_Da"), "SelfService"))
    return flag(obs("SelfService is enabled, but it should be disabled.")) if b is True else ok()


def eval_C023(ctx: DatabricksContext) -> ControlResult:
    """MinBid: exactly 0.02 → OK, anything else → FLAG"""
    sh, df, col, miss = seller_param_row7(ctx, "MinBid", "MinBid|minbid|min_bid", no_data_flag=True)
    if miss:
        return miss
    v = as_float(df.iloc[0][col])
    if v is None:
        return flag(note_data_missing(sh or expected_tab_label("40_Seller_Parameter_Insights_Da"), "MinBid"))
    return ok() if abs(v - 0.02) < 1e-9 else flag(obs(f"MinBid is not at the expected value. Current value is {v}; expected 0.02."))


def eval_C024(ctx: DatabricksContext) -> ControlResult:
    """MaxConversionRate: exactly 25.00 → OK, anything else → FLAG"""
    sh, df, col, miss = seller_param_row7(ctx, "MaxConversionRate", "MaxConversionRate|maxconversionrate|max_conversion_rate", no_data_flag=True)
    if miss:
        return miss
    v = as_float(df.iloc[0][col])
    if v is None:
        return flag(note_data_missing(sh or expected_tab_label("40_Seller_Parameter_Insights_Da"), "MaxConversionRate"))
    return ok() if abs(v - 25.00) <= 1e-6 else flag(obs(f"MaxConversionRate is not at the expected value. Current value is {v}; expected 25.00."))


def eval_C025(ctx: DatabricksContext) -> ControlResult:
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
    return flag(obs(f"PromoteKeywordMinClicks and NegateKeywordMinClicks are not aligned with the expected setup. Current values are {v1} and {v2}; both should be 0."))


def eval_C026(ctx: DatabricksContext) -> ControlResult:
    sh, df, col, miss = seller_param_row7(ctx, "BudgetManagement", "BudgetManagement|budgetmanagement|budget_management", no_data_flag=True)
    if miss:
        return miss
    b = as_bool(df.iloc[0][col])
    if b is None:
        return flag(note_data_missing(sh or expected_tab_label("40_Seller_Parameter_Insights_Da"), "BudgetManagement"))
    return ok() if b is True else flag(obs("BudgetManagement is disabled, but it should be enabled."))


def eval_C027(ctx: DatabricksContext) -> ControlResult:
    sh, df, col, miss = seller_param_row7(ctx, "PlacementModifierManagement", "PlacementModifierManagement|placementmodifiermanagement|placement_modifier_management", no_data_flag=True)
    if miss:
        return miss
    b = as_bool(df.iloc[0][col])
    if b is None:
        return flag(note_data_missing(sh or expected_tab_label("40_Seller_Parameter_Insights_Da"), "PlacementModifierManagement"))
    return ok() if b is True else flag(obs("PlacementModifierManagement is disabled, but it should be enabled."))


def eval_C028(ctx: DatabricksContext) -> ControlResult:
    sh, df, col, miss = seller_param_row7(ctx, "MktStreamHourlyBidAdjustments", "MktStreamHourlyBidAdjustments|mktstreamhourlybidadjustments|mkt_stream_hourly_bid_adjustments", no_data_flag=True)
    if miss:
        return miss
    b = as_bool(df.iloc[0][col])
    if b is None:
        return flag(note_data_missing(sh or expected_tab_label("40_Seller_Parameter_Insights_Da"), "MktStreamHourlyBidAdjustments"))
    return ok() if b is True else flag(obs("MktStreamHourlyBidAdjustments is disabled, but it should be enabled."))


def eval_C029(ctx: DatabricksContext) -> ControlResult:
    sh, df, col, miss = seller_param_row7(ctx, "AutomaticallyImportCampaigns", "AutomaticallyImportCampaigns|automaticallyimportcampaigns|automatically_import_campaigns", no_data_flag=True)
    if miss:
        return miss
    b = as_bool(df.iloc[0][col])
    if b is None:
        return flag(note_data_missing(sh or expected_tab_label("40_Seller_Parameter_Insights_Da"), "AutomaticallyImportCampaigns"))
    return flag(obs("AutomaticallyImportCampaigns is enabled, but it should be disabled.")) if b is True else ok()


def eval_C030(ctx: DatabricksContext) -> ControlResult:
    sh, df, col, miss = seller_param_row7(ctx, "StopAudienceAutoLink", "StopAudienceAutoLink|stopaudienceautolink|stop_audience_auto_link", no_data_flag=True)
    if miss:
        return miss
    b = as_bool(df.iloc[0][col])
    if b is None:
        return flag(note_data_missing(sh or expected_tab_label("40_Seller_Parameter_Insights_Da"), "StopAudienceAutoLink"))
    return flag(obs("StopAudienceAutoLink is enabled, but it should be disabled.")) if b is True else ok()


def eval_C031(ctx: DatabricksContext) -> ControlResult:
    sh, df, col, miss = seller_param_row7(ctx, "IsB2bPlacementManagement", "IsB2bPlacementManagement|isb2bplacementmanagement|is_b2b_placement_management|b2bplacementmanagement", no_data_flag=True)
    if miss:
        return miss
    b = as_bool(df.iloc[0][col])
    if b is None:
        return flag(note_data_missing(sh or expected_tab_label("40_Seller_Parameter_Insights_Da"), "IsB2bPlacementManagement"))
    return ok() if b is True else flag(obs("IsB2bPlacementManagement is disabled, but it should be enabled."))


def eval_C032(ctx: DatabricksContext) -> ControlResult:
    sh, df, col, miss = seller_param_row7(ctx, "HasDisplayPromote", "HasDisplayPromote|hasdisplaypromote|has_display_promote", no_data_flag=True)
    if miss:
        return miss
    b = as_bool(df.iloc[0][col])
    if b is None:
        return flag(note_data_missing(sh or expected_tab_label("40_Seller_Parameter_Insights_Da"), "HasDisplayPromote"))
    return ok() if b is True else flag(obs("HasDisplayPromote is disabled, but it should be enabled."))


def eval_C033(ctx: DatabricksContext) -> ControlResult:
    sh, df, col, miss = seller_param_row7(ctx, "ChangeSBV", "ChangeSBV|changesbv|change_sbv", no_data_flag=True)
    if miss:
        return miss
    b = as_bool(df.iloc[0][col])
    if b is None:
        return flag(note_data_missing(sh or expected_tab_label("40_Seller_Parameter_Insights_Da"), "ChangeSBV"))
    return ok() if b is True else flag(obs("ChangeSBV is disabled, but it should be enabled."))


# ---- C034: RBO Configuration ----
def eval_C034(ctx: DatabricksContext) -> ControlResult:
    sh, df = ds(ctx, "RBO_CONFIG", "33_RBO_Configuration_Insights")
    if df is None or df.empty:
        return ok()

    tab = sh or expected_tab_label("33_RBO_Configuration_Insights")

    if df.shape[1] < 6:
        return flag(note_data_missing(tab, "Data_Type/Setting_Value/Rule_Name"))

    data_type_col = df.columns[0]
    setting_val_col = df.columns[3]
    rule_name_col = df.columns[5]

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
    return flag(obs(f"{n} RBO rule(s) were found in {tab}. Examples: {ex_txt}."))


# ---- C035: SPT Coverage ----
def eval_C035(ctx: DatabricksContext) -> ControlResult:
    """
    Categories qualify for SPT requirement only if:
      - AsinCount >= 30 AND TotalSalesPct >= 5%
    """
    sh10, df10 = ds(ctx, "CAMPAIGNS_BY_TYPE", "10_Campaigns_Grouped_by_QT_Camp")
    sh18, df18 = ds(ctx, "PERF_BY_CATEGORY", "18_Performance_by_Category")
    if (df10 is None or df10.empty) or (df18 is None or df18.empty):
        return ok()

    asin_cnt = find_col(df18, ["asincount", "asin_count", "asin count"])
    if not asin_cnt:
        asin_cnt = get_col_by_letter(df18, "B")
    sales_pct = find_col(df18, ["totalsalespct", "total_sales_pct", "TotalSalesPct"])
    if not sales_pct:
        sales_pct = get_col_by_letter(df18, "J")
    if not asin_cnt or not sales_pct:
        return ok()

    tmp18 = df18.copy()
    tmp18["_asin_cnt"] = tmp18[asin_cnt].apply(as_int).fillna(0)
    tmp18["_sales_pct"] = tmp18[sales_pct].apply(as_float).fillna(0.0)
    tmp18["_sales_pct_norm"] = tmp18["_sales_pct"].apply(lambda v: _normalize_pct(v) if v is not None else 0.0)

    req = int(((tmp18["_asin_cnt"] >= 30) & (tmp18["_sales_pct_norm"] >= 5.0)).sum())

    subtype = find_col(df10, ["campaignsubtype"])
    campaigns = find_col(df10, ["campaigns"])
    if not subtype or not campaigns:
        return ok()

    rows = df10[df10[subtype].astype(str).str.strip().str.upper() == "SPT"]
    spt = int(as_int(rows[campaigns].dropna().iloc[0]) or 0) if not rows.empty else 0

    if spt == 0:
        return flag(obs("No SPT campaigns were found."))
    if req > 0 and spt < req:
        return partial(obs(f"SPT campaign coverage is below target. There are {spt} SPT campaigns versus {req} required categories."))
    return ok()


# ---- C036: CatchAll vs WATM ----
def eval_C036(ctx: DatabricksContext) -> ControlResult:
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
                return partial(obs("CatchAll is present even though it is not required, and WATM is missing."))
            return ok()
        return flag(obs("Neither WATM nor CatchAll was detected."))

    if catchall:
        return ok()
    if watm > 0 and not catchall:
        return partial(obs("CatchAll is required, but only WATM was detected."))
    return flag(obs("CatchAll is required but missing, and no WATM campaign was detected."))


# ---- C037: ATM Coverage — Top Sellers ----
def eval_C037(ctx: DatabricksContext) -> ControlResult:
    """
    FLAG only: Tier 30 ASINs with >1.5 orders/day (>=45 orders in 30d window) and zero ATM spend.
    Tier column is string "TIER 30" (case-insensitive).
    """
    sh, df = ds(ctx, "CAMPAIGN_PERF", "14_Campaign_Performance_by_Adve")
    if df is None or df.empty:
        return flag(note_data_missing(expected_tab_label("14_Campaign_Performance_by_Adve"), "ASIN/Orders/ATM_Spend/Tier"))

    asin = find_col(df, ["asin"])
    orders = find_col(df, ["orders", "purchases"])
    tier = find_col(df, ["tier"])
    atm_spend = find_col(df, ["atm_spend", "atm spend", "atmspend", "atm"])

    if not asin or not orders or not atm_spend or not tier:
        missing = []
        if not asin: missing.append("ASIN")
        if not orders: missing.append("Orders")
        if not atm_spend: missing.append("ATM_Spend")
        if not tier: missing.append("Tier")
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
    tmp["_tier"] = tmp[tier].astype(str).str.strip().str.upper()

    # FLAG: Tier 30 ASINs with >1.5 orders/day and zero ATM spend
    cond_flag = (
        (tmp["_tier"] == "TIER 30") &
        (tmp["_orders_per_day"] > 1.5) &
        (tmp["_atm"] == 0.0)
    )
    if cond_flag.any():
        n = int(cond_flag.sum())
        return flag(obs(f"{n} TIER 30 ASIN(s) with more than 1.5 orders per day have no ATM spend assigned."))

    return ok()


# ---- C038: ATM Catalog Coverage ----
def eval_C038(ctx: DatabricksContext) -> ControlResult:
    """
    Checks ASINs with <1.5 orders/day that have ATM spend active.
    % = such ASINs / total ASINs in tab 14.
    >20% → FLAG, 10-20% → PARTIAL, <10% → OK.
    """
    sh, df = ds(ctx, "CAMPAIGN_PERF", "14_Campaign_Performance_by_Adve")
    if df is None or df.empty:
        return ok()

    asin = find_col(df, ["asin"])
    orders = find_col(df, ["orders", "purchases"])
    atm_spend = find_col(df, ["atm_spend", "atm spend", "atmspend", "atm"])

    if not asin or not orders or not atm_spend:
        return ok()

    days = getattr(ctx, "window_days", None) or 30

    tmp = df.copy()
    tmp["_asin"] = tmp[asin].astype(str).fillna("").str.strip()
    tmp = tmp[tmp["_asin"] != ""].copy()
    if tmp.empty:
        return ok()

    total_asins = len(tmp)

    tmp["_orders"] = tmp[orders].apply(as_float).fillna(0.0)
    tmp["_orders_per_day"] = tmp["_orders"] / float(days)
    tmp["_atm"] = tmp[atm_spend].apply(as_float).fillna(0.0)

    # ASINs with <1.5 orders/day that still have ATM spend active
    cond = (tmp["_orders_per_day"] < 1.5) & (tmp["_atm"] > 0.0)
    affected = int(cond.sum())

    if total_asins == 0 or affected == 0:
        return ok()

    pct = (affected / total_asins) * 100.0

    if pct > 20.0:
        return flag(obs(f"{affected} of {total_asins} ASINs ({pct:.1f}%) with fewer than 1.5 orders per day have ATM spend active, above the 20% threshold."))
    if pct >= 10.0:
        return partial(obs(f"{affected} of {total_asins} ASINs ({pct:.1f}%) with fewer than 1.5 orders per day have ATM spend active, between the 10%–20% caution range."))
    return ok()


# ---- C039: Branded vs Non-Branded Mix ----
def eval_C039(ctx: DatabricksContext) -> ControlResult:
    """
    Checks branded spend % (col B / ad_spend) and branded sales % (col D / ad_sales).
    Both calculated: Branded row value / sum of all values.
    Thresholds: <1% or >35% → FLAG, 1-5% or 20-35% → PARTIAL, 5-20% → OK.
    Worst result between spend and sales wins.
    """
    sh, df = ds(ctx, "SEARCH_TERMS_BY_CATEGORY", "12_Search_Terms_by_Category")
    if df is None or df.empty:
        return flag(note_data_missing(expected_tab_label("12_Search_Terms_by_Category"), "KeywordCategory/ad_spend/ad_sales"))

    cat_col = find_col(df, ["keywordcategory", "category"])
    spend_col = find_col(df, ["ad_spend"]) or get_col_by_letter(df, "B")
    sales_col = find_col(df, ["ad_sales"]) or get_col_by_letter(df, "D")

    if not cat_col or not spend_col or not sales_col:
        return flag(note_data_missing(sh or expected_tab_label("12_Search_Terms_by_Category"), "KeywordCategory/ad_spend/ad_sales"))

    # Find Branded row dynamically by col A
    branded_mask = df[cat_col].astype(str).str.strip().str.lower() == "branded"
    if not branded_mask.any():
        return flag(obs("Branded row is missing from Search Terms by Category."))

    branded_row = df[branded_mask].iloc[0]

    def _calc_pct(col: str) -> Optional[float]:
        total = df[col].apply(as_float).fillna(0.0).sum()
        if total == 0:
            return None
        branded_val = as_float(branded_row[col])
        if branded_val is None:
            return None
        return (branded_val / total) * 100.0

    spend_pct = _calc_pct(spend_col)
    sales_pct = _calc_pct(sales_col)

    def _classify(pct: Optional[float], label: str) -> Optional[ControlResult]:
        if pct is None:
            return None
        if pct < 1.0 or pct > 35.0:
            return flag(obs(f"Branded {label} share is outside the acceptable range at {pct:.2f}%. Expected 1%–35%."))
        if (1.0 <= pct < 5.0) or (20.0 < pct <= 35.0):
            return partial(obs(f"Branded {label} share is in the caution range at {pct:.2f}%. Preferred range is 5%–20%."))
        return ok()

    spend_result = _classify(spend_pct, "spend")
    sales_result = _classify(sales_pct, "sales")

    # Worst result wins: FLAG > PARTIAL > OK
    results = [r for r in [spend_result, sales_result] if r is not None]
    if not results:
        return ok()

    statuses = [r.status for r in results]
    if cfg.STATUS_FLAG in statuses:
        return next(r for r in results if r.status == cfg.STATUS_FLAG)
    if cfg.STATUS_PARTIAL in statuses:
        return next(r for r in results if r.status == cfg.STATUS_PARTIAL)
    return ok()


# ---- C040: SB Spend Share ----
def eval_C040(ctx: DatabricksContext) -> ControlResult:
    sh, df = ds(ctx, "CAMPAIGNS_BY_AMAZON", "09_Campaigns_Grouped_by_Amazon_")
    if df is None or df.empty:
        return flag(note_data_missing(expected_tab_label("09_Campaigns_Grouped_by_Amazon_"), "Campaign_Type/Perc_Spend"))
    ctype = find_col(df, ["campaign_type", "type"])
    pct = find_col(df, ["perc_spend", "pct_spend", "spend_pct", "spend pct"])
    if not ctype or not pct:
        return flag(note_data_missing(sh or expected_tab_label("09_Campaigns_Grouped_by_Amazon_"), "Campaign_Type/Perc_Spend"))
    rows = df[df[ctype].astype(str).str.strip().str.lower() == "sponsored brands"]
    if rows.empty:
        return flag(obs("Sponsored Brands row is missing."))
    v = as_float(rows[pct].dropna().iloc[0]) if not rows[pct].dropna().empty else None
    if v is None:
        return flag(obs("Sponsored Brands spend share is missing."))
    v = _normalize_pct(v)
    return flag(obs(f"Sponsored Brands spend share is outside the expected range at {v:.2f}%. Expected range is 1%–25%.")) if (v < 1.0 or v > 25.0) else ok()


# ---- C041: SB + SBV Combined Spend Share ----
def eval_C041(ctx: DatabricksContext) -> ControlResult:
    """
    Sum Sponsored Brands + Sponsored Brand Video spend, divide by total spend (sum col H / Spend).
    >40% → FLAG, 30-40% → PARTIAL, <=30% → OK.
    """
    sh, df = ds(ctx, "CAMPAIGNS_BY_AMAZON", "09_Campaigns_Grouped_by_Amazon_")
    if df is None or df.empty:
        return ok()

    ctype = find_col(df, ["campaign_type", "type"])
    spend_col = find_col(df, ["spend"]) or get_col_by_letter(df, "H")
    if not ctype or not spend_col:
        return ok()

    tmp = df.copy()
    tmp["_type"] = tmp[ctype].astype(str).str.strip().str.lower()
    tmp["_spend"] = tmp[spend_col].apply(as_float).fillna(0.0)

    total_spend = tmp["_spend"].sum()
    if total_spend == 0:
        return ok()

    sb_spend = tmp[tmp["_type"] == "sponsored brands"]["_spend"].sum()
    sbv_spend = tmp[tmp["_type"] == "sponsored brand video"]["_spend"].sum()
    combined = sb_spend + sbv_spend

    pct = (combined / total_spend) * 100.0

    if pct > 40.0:
        return flag(obs(f"Combined SB + SBV spend share is {pct:.2f}%, above the 40% FLAG threshold."))
    if pct > 30.0:
        return partial(obs(f"Combined SB + SBV spend share is {pct:.2f}%, above the 30% PARTIAL threshold."))
    return ok()


# ---- C042: SBV Spend Share Cap ----
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
    return flag(obs(f"{row_label} spend share is too high at {v:.2f}%, above the {cap:.0f}% cap.")) if v > cap else ok()


def eval_C042(ctx: DatabricksContext) -> ControlResult:
    return _eval_optional_spend_share_cap(ctx, "Sponsored Brand Video", cap=20.0)


# ---- C043: SBTV Spend Share Cap ----
def eval_C043(ctx: DatabricksContext) -> ControlResult:
    return _eval_optional_spend_share_cap(ctx, "Sponsored Brand TV Video", cap=10.0)


# ---- C044: SD Defensive Coverage ----
def eval_C044(ctx: DatabricksContext) -> ControlResult:
    """
    Categories qualify only if AsinCount >= 30 AND TotalSalesPct >= 5%.
    """
    sh10, df10 = ds(ctx, "CAMPAIGNS_BY_TYPE", "10_Campaigns_Grouped_by_QT_Camp")
    sh18, df18 = ds(ctx, "PERF_BY_CATEGORY", "18_Performance_by_Category")
    if (df10 is None or df10.empty) or (df18 is None or df18.empty):
        return ok()

    asin_cnt = find_col(df18, ["asincount", "asin_count", "asin count"]) or get_col_by_letter(df18, "B")
    sales_pct = find_col(df18, ["totalsalespct", "total_sales_pct", "TotalSalesPct"]) or get_col_by_letter(df18, "J")
    if not asin_cnt or not sales_pct:
        return ok()

    tmp18 = df18.copy()
    tmp18["_asin_cnt"] = tmp18[asin_cnt].apply(as_int).fillna(0)
    tmp18["_sales_pct"] = tmp18[sales_pct].apply(as_float).fillna(0.0)
    tmp18["_sales_pct_norm"] = tmp18["_sales_pct"].apply(lambda v: _normalize_pct(v) if v is not None else 0.0)

    req = int(((tmp18["_asin_cnt"] >= 30) & (tmp18["_sales_pct_norm"] >= 5.0)).sum())

    subtype = find_col(df10, ["campaignsubtype"])
    campaigns = find_col(df10, ["campaigns"])
    if not subtype or not campaigns:
        return ok()
    rows = df10[df10[subtype].astype(str).str.strip().str.upper() == "SD_SPT"]
    sd_spt = int(as_int(rows[campaigns].dropna().iloc[0]) or 0) if not rows.empty else 0

    if sd_spt == 0:
        return flag(obs("No SD_SPT campaigns were found."))
    if req > 0 and sd_spt < req:
        return partial(obs(f"SD_SPT campaign coverage is below target. There are {sd_spt} campaigns versus {req} required categories."))
    return ok()


# ---- C045: SD VCPM Spend Share ----
def eval_C045(ctx: DatabricksContext) -> ControlResult:
    """
    VCPM spend share: 5-10% → PARTIAL, >10% → FLAG, <=5% → OK.
    """
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
    if v > 10.0:
        return flag(obs(f"VCPM spend share is too high at {v:.2f}%, above the 10% threshold."))
    if v > 5.0:
        return partial(obs(f"VCPM spend share is elevated at {v:.2f}%, between the 5%–10% caution range."))
    return ok()


# ---- C046: SD VCPM Sales Share ----
def eval_C046(ctx: DatabricksContext) -> ControlResult:
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
    return flag(obs(f"VCPM sales share is too high at {share:.2f}%, above the 20% threshold.")) if share > 20.0 else ok()


# ---- C047: Out of Budget ----
def eval_C047(ctx: DatabricksContext) -> ControlResult:
    """
    Count distinct OOB days in last 30 days from ref_date.
    >=3 days → FLAG, 2 days → PARTIAL, <=1 day → OK.
    """
    sh, df = ds(ctx, "ACCOUNT_OOB", "36_Account_Out_of_Budget")
    if df is None or df.empty:
        return ok()

    dt = find_col(df, ["reportdate", "date", "day"]) or get_col_by_letter(df, "A")
    if not dt:
        return ok()

    if not ctx.ref_date:
        return ok()

    all_dts = pd.to_datetime(df[dt], errors="coerce").dropna().dt.normalize()
    if all_dts.empty:
        return ok()

    cutoff = pd.Timestamp(ctx.ref_date - timedelta(days=30))
    recent = all_dts[all_dts >= cutoff]
    uniq = len(set(recent.tolist()))

    if uniq >= 3:
        return flag(obs(f"The account was out of budget on {uniq} distinct days in the last 30 days, meeting the 3-day FLAG threshold."))
    if uniq == 2:
        return partial(obs("The account was out of budget on 2 distinct days in the last 30 days."))
    return ok()


# ---- C048: Cleanup & Relaunch Cadence (Manual) ----
def eval_C048(ctx: DatabricksContext) -> ControlResult:
    return ok("MANUAL CHECK REQUIRED")


# =========================================================
# Registry + runner
# =========================================================
ALL_CONTROL_IDS: List[str] = [
    "C001", "C002", "C003", "C004", "C005", "C006", "C007", "C008",
    "C009", "C010", "C011", "C012", "C013", "C014", "C015", "C016",
    "C017", "C018", "C019", "C020", "C021", "C022", "C023", "C024",
    "C025", "C026", "C027", "C028", "C029", "C030", "C031", "C032",
    "C033", "C034", "C035", "C036", "C037", "C038", "C039", "C040",
    "C041", "C042", "C043", "C044", "C045", "C046", "C047", "C048",
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
}


def evaluate_all(ctx: DatabricksContext) -> Dict[str, ControlResult]:
    """Deterministic: iterate C001–C048 in order and run the registered function per control."""
    results: Dict[str, ControlResult] = {}
    for cid in ALL_CONTROL_IDS:
        fn = CONTROL_REGISTRY.get(cid)
        if not fn:
            results[cid] = ok()
            continue
        try:
            results[cid] = fn(ctx)
        except Exception as e:
            results[cid] = ControlResult(cfg.STATUS_FLAG, f"EXCEPTION in {cid}: {type(e).__name__}: {e}")
    return results
