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
    return f"Data not available: {tab} — column {col} was not found or is empty."


def ok(what: str = "", why: str = "") -> ControlResult:
    return ControlResult(cfg.STATUS_OK, what, why)


def flag(what: str = "", why: str = "") -> ControlResult:
    what = (what or "").strip()
    if not what:
        what = "FLAG triggered — no detail available."
    return ControlResult(cfg.STATUS_FLAG, what, why)


def partial(what: str = "", why: str = "") -> ControlResult:
    return ControlResult(cfg.STATUS_PARTIAL, what, why)


def expected_tab_label(dataset_key: str) -> str:
    prefixes = getattr(cfg, "TAB_CANDIDATES", {}).get(dataset_key, [])
    return prefixes[0] if prefixes else dataset_key


# =========================================================
# Helpers (stable)
# =========================================================
def norm(s: str) -> str:
    return (
        str(s).strip().lower()
        .replace("\n", " ").replace("\r", " ").replace("\t", " ").replace(" ", "_")
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


def _normalize_pct(v: float) -> float:
    return v * 100.0 if 0 < v <= 1.0 else float(v)


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


# =========================================================
# Portfolio threshold helper
# =========================================================
def _portfolio_threshold_check(
    ctx: DatabricksContext,
    col_candidates: List[str],
    condition_label: str,
    why_text: str,
) -> ControlResult:
    sh, df = ds(ctx, "PORTFOLIO_INSIGHTS", "25_Portfolio_Insights_and_Confi")
    if df is None or df.empty:
        return ok("No portfolio data found. Nothing to evaluate.", why_text)

    is_managed = find_col(df, ["ismanaged"])
    if not is_managed:
        return ok("Portfolio data found but no managed/unmanaged flag detected.", why_text)

    elig = df[df[is_managed].apply(as_bool) == True].copy()  # noqa: E712
    total_managed = len(elig)
    if total_managed == 0:
        return ok("No managed portfolios found. Nothing to evaluate.", why_text)

    col = find_col(elig, col_candidates)
    if not col:
        return ok(f"No {condition_label} column found in portfolio data.", why_text)

    active_count = int((elig[col].apply(as_bool) == True).sum())  # noqa: E712
    pct = (active_count / total_managed) * 100.0

    if pct > 50.0:
        return flag(
            f"{active_count} of {total_managed} managed portfolios ({pct:.1f}%) have {condition_label} enabled — above the 50% threshold.",
            why_text,
        )
    if pct >= 25.0:
        return partial(
            f"{active_count} of {total_managed} managed portfolios ({pct:.1f}%) have {condition_label} enabled — in the 25%–50% caution range.",
            why_text,
        )
    return ok(
        f"{active_count} of {total_managed} managed portfolios ({pct:.1f}%) have {condition_label} enabled — within the acceptable range.",
        why_text,
    )


# =========================================================
# Controls C001–C048
# =========================================================

# ---- C001/C002/C003: ACoS change governance ----
def _load_acos_changes(ctx: DatabricksContext) -> Tuple[Optional[str], Optional[pd.DataFrame], Optional[str], Optional[str]]:
    sh, df = ds(ctx, "ACOS_CHANGES", "24_Account_ACoS_Changes_History")
    if df is None or df.empty:
        return sh, df, None, None
    col_date = find_col(df, ["change_date", "change date", "changedate"])
    col_val = find_col(df, ["iacos_percent", "iacos percent", "iacos_percent_", "iacos"])
    return sh, df, col_date, col_val


def eval_C001(ctx: DatabricksContext) -> ControlResult:
    WHY = "ACoS changes that are too close together stop the system from learning. The minimum gap between changes is 14 days."
    sh, df, col_date, _ = _load_acos_changes(ctx)
    if df is None or df.empty:
        return ok("No ACoS change history found. Nothing to evaluate.", WHY)
    if not ctx.ref_date or not col_date:
        return ok("ACoS change history found but reference date or date column is missing.", WHY)

    tmp = df.copy()
    tmp["_dt"] = pd.to_datetime(tmp[col_date], errors="coerce")
    tmp = tmp[tmp["_dt"].notna()].copy()
    if tmp.empty:
        return ok("ACoS change history found but no valid dates were detected.", WHY)

    def min_gap_days(window_days: int) -> Optional[int]:
        cutoff = pd.Timestamp(ctx.ref_date - timedelta(days=window_days))
        w = tmp[tmp["_dt"] >= cutoff].copy()
        if len(w.index) < 2:
            return None
        dts_sorted = w["_dt"].sort_values().reset_index(drop=True)
        gaps = dts_sorted.diff().dt.days.dropna()
        return int(gaps.min()) if not gaps.empty else None

    g90 = min_gap_days(90)
    if g90 is not None and g90 < 14:
        return flag(f"ACoS changes were made too close together in the last 90 days. Shortest gap was {g90} days — minimum required is 14 days.", WHY)
    g180 = min_gap_days(180)
    if g180 is not None and g180 < 14:
        return partial(f"ACoS changes were relatively close together in the last 180 days. Shortest gap was {g180} days — minimum required is 14 days.", WHY)
    return ok("ACoS change spacing is within the 14-day minimum across the last 90 and 180 days.", WHY)


def eval_C002(ctx: DatabricksContext) -> ControlResult:
    WHY = "Too many ACoS changes in a short window prevent the system from stabilizing. The limit is 5 changes in 90 days."
    sh, df, col_date, _ = _load_acos_changes(ctx)
    if df is None or df.empty:
        return ok("No ACoS change history found. Nothing to evaluate.", WHY)
    if not ctx.ref_date or not col_date:
        return ok("ACoS change history found but reference date or date column is missing.", WHY)

    dts = pd.to_datetime(df[col_date], errors="coerce").dropna()
    if dts.empty:
        return ok("ACoS change history found but no valid dates were detected.", WHY)
    cutoff_90 = pd.Timestamp(ctx.ref_date - timedelta(days=90))
    changes_90 = int((dts >= cutoff_90).sum())
    if changes_90 > 5:
        return flag(f"{changes_90} ACoS changes were made in the last 90 days — above the limit of 5.", WHY)
    return ok(f"{changes_90} ACoS changes in the last 90 days — within the limit of 5.", WHY)


def eval_C003(ctx: DatabricksContext) -> ControlResult:
    WHY = "Each ACoS adjustment should be meaningful but not too aggressive. Changes smaller than 5% or larger than 25% relative to the previous value are outside the acceptable range."
    sh, df, col_date, col_val = _load_acos_changes(ctx)
    if df is None or df.empty:
        return ok("No ACoS change history found. Nothing to evaluate.", WHY)
    if not ctx.ref_date or not col_date or not col_val:
        return ok("ACoS change history found but date or value column is missing.", WHY)

    tmp = df.copy()
    tmp["_dt"] = pd.to_datetime(tmp[col_date], errors="coerce")
    tmp["_v"] = tmp[col_val].apply(as_float)
    tmp = tmp[tmp["_dt"].notna() & tmp["_v"].notna()].copy()
    if tmp.empty or len(tmp.index) < 2:
        return ok("Not enough ACoS change records to evaluate magnitude.", WHY)

    cutoff_90 = pd.Timestamp(ctx.ref_date - timedelta(days=90))
    tmp = tmp[tmp["_dt"] >= cutoff_90].sort_values("_dt")
    vals = [_normalize_pct(v) for v in tmp["_v"].tolist()]
    if len(vals) < 2:
        return ok("Only one ACoS change in the last 90 days — magnitude check not applicable.", WHY)

    for old, new in zip(vals[:-1], vals[1:]):
        if old == 0:
            return flag("The previous ACoS value was 0 — relative change magnitude could not be calculated.", WHY)
        mag = abs((new - old) / old) * 100.0
        if mag < 5.0 or mag > 25.0:
            return flag(f"ACoS change magnitude was {mag:.1f}% — outside the acceptable 5%–25% range.", WHY)
    return ok("All ACoS change magnitudes in the last 90 days are within the 5%–25% acceptable range.", WHY)


# ---- C004/C005: Seller params — QuartileFactor / CurrentEpisolon ----
def eval_C004(ctx: DatabricksContext) -> ControlResult:
    WHY = "A non-default QuartileFactor changes how the system behaves across the portfolio and can cause unpredictable bid adjustments."
    sh, df, col, miss = seller_param_row7(ctx, "QuartileFactor", "QuartileFactor|quartile_factor|quartilefactor", no_data_flag=True)
    if miss:
        return ControlResult(miss.status, miss.what, WHY)
    v = as_float(df.iloc[0][col])
    if v is not None and abs(v - 1.0) < 1e-9:
        return ok(f"QuartileFactor is set to {v} — matches the expected value of 1.0.", WHY)
    return flag(f"QuartileFactor is {v} — expected 1.0. A non-default value changes system behavior.", WHY)


def eval_C005(ctx: DatabricksContext) -> ControlResult:
    WHY = "A non-default epsilon value overrides default VAM baseline behavior and can shift bid decisions across the account."
    sh, df, col, miss = seller_param_row7(ctx, "CurrentEpisolon", "CurrentEpisolon|CurrentEpsilon|currentepsilon|currentepisolon|current_epsilon", no_data_flag=True)
    if miss:
        return ControlResult(miss.status, miss.what, WHY)
    v = as_float(df.iloc[0][col])
    if v is not None and abs(v - 1.0) < 1e-9:
        return ok(f"CurrentEpisolon is set to {v} — matches the expected value of 1.0.", WHY)
    return flag(f"CurrentEpisolon is {v} — expected 1.0. A non-default value overrides VAM baseline behavior.", WHY)


# ---- C006/C007: Presence-based ACoS overrides ----
def eval_C006(ctx: DatabricksContext) -> ControlResult:
    WHY = "Product-level ACoS targets override account-level governance. Each override needs to be intentional and documented."
    sh, df = ds(ctx, "PRODUCT_LEVEL_ACOS", "34_Product_Level_ACoS")
    if df is None or df.empty:
        return ok("No product-level ACoS overrides found.", WHY)

    asin_col = find_col(df, ["child_product", "asin", "product"])
    acos_col = find_col(df, ["acos_percent", "acos"])
    n = len(df)

    if asin_col and acos_col:
        items = []
        for _, row in df.head(3).iterrows():
            asin = _clean_cell_to_str(row[asin_col])
            acos_val = as_float(row[acos_col])
            if asin and acos_val is not None:
                items.append(f"{asin} @ {acos_val:.0f}%")
        if items:
            suffix = f" (showing first {len(items)} of {n})" if n > 3 else ""
            return flag(f"{n} product-level ACoS override(s) detected{suffix}: {', '.join(items)}.", WHY)

    return flag(f"{n} product-level ACoS override(s) detected. Check tab 34 for details.", WHY)


def eval_C007(ctx: DatabricksContext) -> ControlResult:
    WHY = "Campaign-level ACoS targets create inconsistency across the portfolio. Each override needs to be intentional and documented."
    sh, df = ds(ctx, "CAMPAIGN_LEVEL_ACOS", "35_Campaign_Level_ACoS")
    if df is None or df.empty:
        return ok("No campaign-level ACoS overrides found.", WHY)

    camp_col = find_col(df, ["campaign_name", "campaignname", "campaign"])
    acos_col = find_col(df, ["acos_percent", "acos"])
    n = len(df)

    if camp_col and acos_col:
        items = []
        for _, row in df.head(3).iterrows():
            name = _clean_cell_to_str(row[camp_col])
            acos_val = as_float(row[acos_col])
            if name and acos_val is not None:
                items.append(f"{name} @ {acos_val:.0f}%")
        if items:
            suffix = f" (showing first {len(items)} of {n})" if n > 3 else ""
            return flag(f"{n} campaign-level ACoS override(s) detected{suffix}: {', '.join(items)}.", WHY)

    return flag(f"{n} campaign-level ACoS override(s) detected. Check tab 35 for details.", WHY)


# ---- C008: Timeframe Boost ----
def eval_C008(ctx: DatabricksContext) -> ControlResult:
    WHY = "An active timeframe boost distorts placement bidding and learning. Boosts should only be active during the intended promotional window."
    sh, df = ds(ctx, "TIMEFRAME_BOOST", "27_Timeframe_Boost")
    if df is None or df.empty:
        return ok("No timeframe boost records found.", WHY)

    status_col = find_col(df, ["status", "Status", "status_name", "statusname"])
    if not status_col:
        return flag(note_data_missing(sh or expected_tab_label("27_Timeframe_Boost"), "Status"), WHY)

    tmp = df.copy()
    tmp["_status"] = tmp[status_col].astype(str).fillna("").str.strip().str.lower()
    tmp = tmp[tmp["_status"] != ""]
    if tmp.empty:
        return ok("Timeframe boost tab found but no valid status rows detected.", WHY)

    if (tmp["_status"] != "expired").any():
        active_rows = tmp[tmp["_status"] != "expired"].copy()
        n = len(active_rows)
        asin_col = find_col(active_rows, ["asin"])
        end_col = find_col(active_rows, ["enddate", "end_date", "EndDate"])
        items = []
        for _, row in active_rows.head(3).iterrows():
            asin = _clean_cell_to_str(row[asin_col]) if asin_col else ""
            end = ""
            if end_col:
                end_val = pd.to_datetime(row[end_col], errors="coerce")
                if not pd.isna(end_val):
                    end = f" ends {end_val.strftime('%Y-%m-%d')}"
            if asin:
                items.append(f"{asin}{end}")
        if items:
            suffix = f" (showing first {len(items)} of {n})" if n > 3 else ""
            return flag(f"{n} timeframe boost(s) still active{suffix}: {', '.join(items)}.", WHY)
        return flag(f"{n} timeframe boost row(s) are still active and not marked as expired.", WHY)
    return ok("All timeframe boost records are expired. No active boosts detected.", WHY)


# ---- C009/C010: Negative Keywords ----
_NEGATIVE_EXCEPTIONS = [
    "deal", "deals", "discount", "black friday",
    "cyber monday", "prime day", "holiday",
]


def _is_exception_negative(term: str) -> bool:
    t = _clean_cell_to_str(term).lower()
    if not t:
        return False
    return any(k in t for k in _NEGATIVE_EXCEPTIONS)


def eval_C009(ctx: DatabricksContext) -> ControlResult:
    WHY = "Account-level negative keywords block traffic for all products. A keyword added by mistake can silently reduce impressions across the whole account."
    sh, df = ds(ctx, "NEGATIVES", "29_Negative_Keywords__Global")
    if df is None or df.empty:
        return ok("No negative keyword data found. Nothing to evaluate.", WHY)

    neg = find_col(df, ["negative_word", "negative word", "negative", "keyword"])
    prod = find_col(df, ["product", "asin", "targetasin"])
    if not neg:
        return ok("Negative keywords tab found but keyword column not detected.", WHY)

    tmp = df.copy()
    tmp["_neg"] = tmp[neg].apply(_clean_cell_to_str)
    tmp = tmp[tmp["_neg"] != ""]
    if tmp.empty:
        return ok("No negative keywords found in the account.", WHY)

    if prod:
        tmp["_prod"] = tmp[prod].apply(_clean_cell_to_str)
        acct = tmp[tmp["_prod"] == ""].copy()
    else:
        acct = tmp.copy()

    if acct.empty:
        return ok("No account-level negative keywords detected.", WHY)

    non_exc = [x for x in acct["_neg"].tolist() if not _is_exception_negative(x)]
    if non_exc:
        return flag(f"{len(non_exc)} account-level negative keyword(s) found that are not standard exceptions.", WHY)
    return ok(f"Account-level negative keywords found but all are standard exceptions (deals, Prime Day, etc.).", WHY)


def eval_C010(ctx: DatabricksContext) -> ControlResult:
    WHY = "Product-level negative keywords block traffic for specific ASINs. They are hard to track and can create invisible performance issues."
    sh, df = ds(ctx, "NEGATIVES", "29_Negative_Keywords__Global")
    if df is None or df.empty:
        return ok("No negative keyword data found. Nothing to evaluate.", WHY)

    neg = find_col(df, ["negative_word", "negative word", "negative", "keyword"])
    prod = find_col(df, ["product", "asin", "targetasin"])
    if not neg:
        return ok("Negative keywords tab found but keyword column not detected.", WHY)

    tmp = df.copy()
    tmp["_neg"] = tmp[neg].apply(_clean_cell_to_str)
    tmp = tmp[tmp["_neg"] != ""]
    if tmp.empty:
        return ok("No negative keywords found in the account.", WHY)

    if not prod:
        return ok("No product column found — product-level negatives cannot be evaluated.", WHY)

    tmp["_prod"] = tmp[prod].apply(_clean_cell_to_str)
    prod_df = tmp[tmp["_prod"] != ""].copy()
    if prod_df.empty:
        return ok("No product-level negative keywords detected.", WHY)

    non_exc = [x for x in prod_df["_neg"].tolist() if not _is_exception_negative(x)]
    if non_exc:
        return flag(f"{len(non_exc)} product-level negative keyword(s) found that are not standard exceptions.", WHY)
    return ok("Product-level negative keywords found but all are standard exceptions.", WHY)


# ---- C011: Product Tag Completeness ----
def eval_C011(ctx: DatabricksContext) -> ControlResult:
    WHY = "Missing tags make it impossible to segment and analyze the portfolio correctly. Tags are required for reliable reporting and strategy decisions."
    sh, df = ds(ctx, "CAMPAIGN_PERF", "14_Campaign_Performance_by_Adve")
    if df is None or df.empty:
        return flag(note_data_missing(expected_tab_label("14_Campaign_Performance_by_Adve"), "ASIN/Tags"), WHY)

    asin_col = find_col(df, ["asin"])
    if not asin_col:
        return flag(note_data_missing(sh or expected_tab_label("14_Campaign_Performance_by_Adve"), "ASIN"), WHY)

    tag_cols = []
    for t in ["tag1", "tag2", "tag3", "tag4", "tag5"]:
        c = find_col(df, [t])
        if c:
            tag_cols.append(c)
    if not tag_cols:
        return flag(note_data_missing(sh or expected_tab_label("14_Campaign_Performance_by_Adve"), "Tag1-Tag5"), WHY)

    active = [c for c in tag_cols if (df[c].astype(str).fillna("").str.strip() != "").any()]

    tag1_col = find_col(df, ["tag1"])
    if tag1_col:
        asin_has_values = (df[asin_col].astype(str).fillna("").str.strip() != "").any()
        tag1_has_values = (df[tag1_col].astype(str).fillna("").str.strip() != "").any()
        if asin_has_values and not tag1_has_values:
            return flag("Tag1 column exists but no ASINs have a Tag1 value assigned.", WHY)

    if not active:
        return ok("No active tag columns detected — tag completeness check not applicable.", WHY)

    elig = df[df[asin_col].astype(str).fillna("").str.strip() != ""].copy()
    if elig.empty:
        return ok("No ASINs with data found for tag evaluation.", WHY)

    total = len(elig.index) * len(active)
    missing = 0
    for c in active:
        missing += (elig[c].astype(str).fillna("").str.strip() == "").sum()
    missing_pct = (missing / total) * 100.0 if total else 0.0

    if missing_pct >= 25.0:
        return flag(f"{missing_pct:.1f}% of tag fields are empty across the portfolio — above the 25% threshold.", WHY)
    if missing_pct >= 10.0:
        return partial(f"{missing_pct:.1f}% of tag fields are empty across the portfolio — between the 10%–25% caution range.", WHY)
    return ok(f"{missing_pct:.1f}% of tag fields are empty — within the acceptable range.", WHY)


# ---- C012/C013: Branded/Competitor terms ----
def eval_C012(ctx: DatabricksContext) -> ControlResult:
    WHY = "At least one branded term must be present. Without branded keywords, the account has no defensive coverage for its own brand search."
    sh, df = ds(ctx, "BRAND_COMP_TERMS", "30_Branded_and_Competitor_Terms")
    if df is None or df.empty or len(df.index) < 1:
        return flag(note_data_missing(expected_tab_label("30_Branded_and_Competitor_Terms"), "Row 7"), WHY)
    total_terms = find_col(df, ["total_terms", "total terms", "terms", "count", "keyword_count", "total"])
    if not total_terms:
        return flag(note_data_missing(sh or expected_tab_label("30_Branded_and_Competitor_Terms"), "Total_Terms"), WHY)
    n = as_int(df.iloc[0][total_terms])
    if n is None:
        return flag(note_data_missing(sh or expected_tab_label("30_Branded_and_Competitor_Terms"), "Total_Terms"), WHY)
    if n >= 1:
        return ok(f"{n} branded term(s) listed — meets the minimum requirement of 1.", WHY)
    return flag(f"Branded terms count is {n} — at least 1 is required.", WHY)


def eval_C013(ctx: DatabricksContext) -> ControlResult:
    WHY = "Competitor terms are important for market share defense and growth. At least 3 competitor terms are recommended."
    sh, df = ds(ctx, "BRAND_COMP_TERMS", "30_Branded_and_Competitor_Terms")
    if df is None or df.empty or len(df.index) < 2:
        return flag(note_data_missing(expected_tab_label("30_Branded_and_Competitor_Terms"), "Row 8"), WHY)
    total_terms = find_col(df, ["total_terms", "total terms", "terms", "count", "keyword_count", "total"])
    if not total_terms:
        return flag(note_data_missing(sh or expected_tab_label("30_Branded_and_Competitor_Terms"), "Total_Terms"), WHY)
    n = as_int(df.iloc[1][total_terms])
    if n is None:
        return flag(note_data_missing(sh or expected_tab_label("30_Branded_and_Competitor_Terms"), "Total_Terms (Row 8)"), WHY)
    if n == 0:
        return flag(f"Competitor terms count is 0 — at least 3 are recommended.", WHY)
    if 1 <= n <= 2:
        return partial(f"{n} competitor term(s) listed — below the recommended level of 3.", WHY)
    return ok(f"{n} competitor term(s) listed — meets the recommended level.", WHY)


# ---- C014–C017: Unmanaged (presence-based) ----
def eval_C014(ctx: DatabricksContext) -> ControlResult:
    WHY = "Unmanaged ASINs are excluded from system control. Active exclusions reduce the coverage of the account and can lower overall performance."
    sh, df = ds(ctx, "UNMANAGED_ASIN", "26_Unmanaged_ASIN")
    if df is None or df.empty:
        return ok("No unmanaged ASIN records found.", WHY)
    if not getattr(ctx, "ref_date", None):
        return flag(note_data_missing(sh or expected_tab_label("26_Unmanaged_ASIN"), "Downloaded timestamp (ref_date)"), WHY)
    if df.shape[1] < 5:
        return flag(note_data_missing(sh or expected_tab_label("26_Unmanaged_ASIN"), "End Date column"), WHY)
    end_col = df.columns[4]
    end_dates = pd.to_datetime(df[end_col], errors="coerce")
    active_mask = end_dates > pd.Timestamp(ctx.ref_date)
    if bool(active_mask.any()):
        active = df[active_mask.values].copy()
        n = len(active)
        asin_col = find_col(active, ["asin", "child_product", "product"])
        items = []
        if asin_col:
            for v in active[asin_col].dropna().head(3).tolist():
                s = _clean_cell_to_str(v)
                if s:
                    items.append(s)
        if items:
            suffix = f" (showing first {len(items)} of {n})" if n > 3 else ""
            return flag(f"{n} unmanaged ASIN(s) are still active{suffix}: {', '.join(items)}.", WHY)
        return flag(f"{n} unmanaged ASIN row(s) are still active based on end date.", WHY)
    return ok("No active unmanaged ASINs detected — all records are expired or empty.", WHY)


def eval_C015(ctx: DatabricksContext) -> ControlResult:
    WHY = "Unmanaged product budgets override system budget controls. Active overrides can cause uncontrolled spend and break budget governance."
    sh, df = ds(ctx, "UNMANAGED_BUDGET", "28_Unmanaged_Budget")
    if df is None or df.empty:
        return ok("No unmanaged product budget records found.", WHY)
    if not getattr(ctx, "ref_date", None):
        return flag(note_data_missing(sh or expected_tab_label("28_Unmanaged_Budget"), "Downloaded timestamp (ref_date)"), WHY)
    if df.shape[1] < 7:
        return flag(note_data_missing(sh or expected_tab_label("28_Unmanaged_Budget"), "End Date column"), WHY)
    end_col = df.columns[6]
    end_dates = pd.to_datetime(df[end_col], errors="coerce")
    active_mask = end_dates > pd.Timestamp(ctx.ref_date)
    if bool(active_mask.any()):
        active = df[active_mask.values].copy()
        n = len(active)
        asin_col = find_col(active, ["asin", "child_product", "product"])
        items = []
        if asin_col:
            for v in active[asin_col].dropna().head(3).tolist():
                s = _clean_cell_to_str(v)
                if s:
                    items.append(s)
        if items:
            suffix = f" (showing first {len(items)} of {n})" if n > 3 else ""
            return flag(f"{n} unmanaged product budget(s) still active{suffix}: {', '.join(items)}.", WHY)
        return flag(f"{n} unmanaged product budget row(s) are still active based on end date.", WHY)
    return ok("No active unmanaged product budgets detected — all records are expired or empty.", WHY)


def eval_C016(ctx: DatabricksContext) -> ControlResult:
    WHY = "Unmanaged campaigns run outside system control. They create inconsistent governance and reduce the reliability of performance data."
    sh, df = ds(ctx, "UNMANAGED_CAMPAIGNS", "31_Unmanaged_campaigns")
    if df is None or df.empty:
        return ok("No unmanaged campaign records found.", WHY)
    if not getattr(ctx, "ref_date", None):
        return flag(note_data_missing(sh or expected_tab_label("31_Unmanaged_campaigns"), "Downloaded timestamp (ref_date)"), WHY)
    if df.shape[1] < 12:
        return flag(note_data_missing(sh or expected_tab_label("31_Unmanaged_campaigns"), "End Date column"), WHY)
    end_col = df.columns[11]
    end_dates = pd.to_datetime(df[end_col], errors="coerce")
    active_mask = end_dates > pd.Timestamp(ctx.ref_date)
    if bool(active_mask.any()):
        active = df[active_mask.values].copy()
        n = len(active)
        camp_col = find_col(active, ["campaignname", "campaign_name", "campaign"])
        items = []
        if camp_col:
            for v in active[camp_col].dropna().head(3).tolist():
                s = _clean_cell_to_str(v)
                if s:
                    items.append(s)
        if items:
            suffix = f" (showing first {len(items)} of {n})" if n > 3 else ""
            return flag(f"{n} unmanaged campaign(s) still active{suffix}: {', '.join(items)}.", WHY)
        return flag(f"{n} unmanaged campaign row(s) are still active based on end date.", WHY)
    return ok("No active unmanaged campaigns detected — all records are expired or empty.", WHY)


def eval_C017(ctx: DatabricksContext) -> ControlResult:
    WHY = "Unmanaged campaign budgets override system controls and can cause uncontrolled spend. Active overrides increase the risk of budget efficiency issues."
    sh, df = ds(ctx, "UNMANAGED_CAMPAIGN_BUDGETS", "32_Unmanaged_Campaigns_Budget_O")
    if df is None or df.empty:
        return ok("No unmanaged campaign budget records found.", WHY)
    if not getattr(ctx, "ref_date", None):
        return flag(note_data_missing(sh or expected_tab_label("32_Unmanaged_Campaigns_Budget_O"), "Downloaded timestamp (ref_date)"), WHY)
    if df.shape[1] < 7:
        return flag(note_data_missing(sh or expected_tab_label("32_Unmanaged_Campaigns_Budget_O"), "End Date column"), WHY)
    end_col = df.columns[6]
    end_dates = pd.to_datetime(df[end_col], errors="coerce")
    active_mask = end_dates > pd.Timestamp(ctx.ref_date)
    if bool(active_mask.any()):
        active = df[active_mask.values].copy()
        n = len(active)
        camp_col = find_col(active, ["campaignname", "campaign_name", "campaign"])
        items = []
        if camp_col:
            for v in active[camp_col].dropna().head(3).tolist():
                s = _clean_cell_to_str(v)
                if s:
                    items.append(s)
        if items:
            suffix = f" (showing first {len(items)} of {n})" if n > 3 else ""
            return flag(f"{n} unmanaged campaign budget(s) still active{suffix}: {', '.join(items)}.", WHY)
        return flag(f"{n} unmanaged campaign budget row(s) are still active based on end date.", WHY)
    return ok("No active unmanaged campaign budgets detected — all records are expired or empty.", WHY)


# ---- C018: ARIS Manual Recommendations ----
def eval_C018(ctx: DatabricksContext) -> ControlResult:
    WHY = "Manual ARIS recommendations break one-on-one campaign governance. Each manual recommendation should be reviewed and resolved."
    sh, df = ds(ctx, "ARIS_MANUAL_RECS", "41_ARIS__Manual_Recomendation")
    if df is None or df.empty:
        return ok("No ARIS manual recommendations found.", WHY)
    n = len(df)
    return flag(f"{n} ARIS manual recommendation(s) are pending — these need to be reviewed and acted on.", WHY)


# ---- C019/C020/C021: Portfolio threshold controls ----
def eval_C019(ctx: DatabricksContext) -> ControlResult:
    WHY = "IsDailyVamBaseline overrides the standard portfolio-level VAM behavior. Having this active on too many portfolios creates inconsistent bid management."
    return _portfolio_threshold_check(ctx, ["isdailyvambaseline", "IsDailyVamBaseline"], "IsDailyVamBaseline", WHY)


def eval_C020(ctx: DatabricksContext) -> ControlResult:
    WHY = "IsTargetACoS overrides the portfolio-level ACoS governance. High usage across portfolios means the account is being managed with inconsistent efficiency targets."
    return _portfolio_threshold_check(ctx, ["istargetacos", "IsTargetAcos", "IsTargetACoS"], "IsTargetACoS", WHY)


def eval_C021(ctx: DatabricksContext) -> ControlResult:
    WHY = "IsBudgetCap limits portfolio-level budget flexibility. High usage across portfolios can constrain delivery and reduce scaling capacity."
    return _portfolio_threshold_check(ctx, ["isbudgetcap", "IsBudgetCap"], "IsBudgetCap", WHY)


# ---- C022–C033: Seller Params ----
def eval_C022(ctx: DatabricksContext) -> ControlResult:
    WHY = "SelfService mode gives the client direct control over campaign settings, bypassing system governance. It should always be disabled."
    sh, df, col, miss = seller_param_row7(ctx, "SelfService", "SelfService|selfservice", no_data_flag=True)
    if miss:
        return ControlResult(miss.status, miss.what, WHY)
    b = as_bool(df.iloc[0][col])
    if b is None:
        return flag(note_data_missing(sh or expected_tab_label("40_Seller_Parameter_Insights_Da"), "SelfService"), WHY)
    if b is True:
        return flag("SelfService is enabled — it should be disabled to maintain system governance.", WHY)
    return ok("SelfService is disabled — correct setting.", WHY)


def eval_C023(ctx: DatabricksContext) -> ControlResult:
    WHY = "MinBid controls the floor for all bids. A value other than 0.02 can prevent the system from placing efficient bids on low-competition terms."
    sh, df, col, miss = seller_param_row7(ctx, "MinBid", "MinBid|minbid|min_bid", no_data_flag=True)
    if miss:
        return ControlResult(miss.status, miss.what, WHY)
    v = as_float(df.iloc[0][col])
    if v is None:
        return flag(note_data_missing(sh or expected_tab_label("40_Seller_Parameter_Insights_Da"), "MinBid"), WHY)
    if abs(v - 0.02) < 1e-9:
        return ok(f"MinBid is {v} — matches the expected value of 0.02.", WHY)
    return flag(f"MinBid is {v} — expected 0.02. This can affect bid floors across all campaigns.", WHY)


def eval_C024(ctx: DatabricksContext) -> ControlResult:
    WHY = "MaxConversionRate caps how high the system estimates CVR. A non-standard value can distort bid calculations and overspend on high-converting terms."
    sh, df, col, miss = seller_param_row7(ctx, "MaxConversionRate", "MaxConversionRate|maxconversionrate|max_conversion_rate", no_data_flag=True)
    if miss:
        return ControlResult(miss.status, miss.what, WHY)
    v = as_float(df.iloc[0][col])
    if v is None:
        return flag(note_data_missing(sh or expected_tab_label("40_Seller_Parameter_Insights_Da"), "MaxConversionRate"), WHY)
    if abs(v - 25.00) <= 1e-6:
        return ok(f"MaxConversionRate is {v} — matches the expected value of 25.00.", WHY)
    return flag(f"MaxConversionRate is {v} — expected 25.00. A non-standard cap can distort bid calculations.", WHY)


def eval_C025(ctx: DatabricksContext) -> ControlResult:
    WHY = "PromoteKeywordMinClicks and NegateKeywordMinClicks control when keywords are automatically promoted or negated. Both must be 0 to keep the system responsive."
    sh, df, col1, miss1 = seller_param_row7(ctx, "PromoteKeywordMinClicks", "PromoteKeywordMinClicks|promotekeywordminclicks|promote_keyword_min_clicks", no_data_flag=True)
    if miss1:
        return ControlResult(miss1.status, miss1.what, WHY)
    sh, df, col2, miss2 = seller_param_row7(ctx, "NegateKeywordMinClicks", "NegateKeywordMinClicks|negatekeywordminclicks|negate_keyword_min_clicks", no_data_flag=True)
    if miss2:
        return ControlResult(miss2.status, miss2.what, WHY)
    v1 = as_float(df.iloc[0][col1])
    v2 = as_float(df.iloc[0][col2])
    if v1 is None or v2 is None:
        return flag(note_data_missing(sh or expected_tab_label("40_Seller_Parameter_Insights_Da"), "Promote/Negate MinClicks"), WHY)
    if abs(v1) < 1e-9 and abs(v2) < 1e-9:
        return ok(f"PromoteKeywordMinClicks is {v1} and NegateKeywordMinClicks is {v2} — both match the expected value of 0.", WHY)
    return flag(f"PromoteKeywordMinClicks is {v1} and NegateKeywordMinClicks is {v2} — both should be 0.", WHY)


def eval_C026(ctx: DatabricksContext) -> ControlResult:
    WHY = "BudgetManagement enables the system to control campaign budgets automatically. When disabled, budgets are not being managed by the platform."
    sh, df, col, miss = seller_param_row7(ctx, "BudgetManagement", "BudgetManagement|budgetmanagement|budget_management", no_data_flag=True)
    if miss:
        return ControlResult(miss.status, miss.what, WHY)
    b = as_bool(df.iloc[0][col])
    if b is None:
        return flag(note_data_missing(sh or expected_tab_label("40_Seller_Parameter_Insights_Da"), "BudgetManagement"), WHY)
    if b is True:
        return ok("BudgetManagement is enabled — correct setting.", WHY)
    return flag("BudgetManagement is disabled — it should be enabled for the system to manage budgets.", WHY)


def eval_C027(ctx: DatabricksContext) -> ControlResult:
    WHY = "PlacementModifierManagement allows the system to optimize placement bids. When disabled, placement adjustments are not being made."
    sh, df, col, miss = seller_param_row7(ctx, "PlacementModifierManagement", "PlacementModifierManagement|placementmodifiermanagement|placement_modifier_management", no_data_flag=True)
    if miss:
        return ControlResult(miss.status, miss.what, WHY)
    b = as_bool(df.iloc[0][col])
    if b is None:
        return flag(note_data_missing(sh or expected_tab_label("40_Seller_Parameter_Insights_Da"), "PlacementModifierManagement"), WHY)
    if b is True:
        return ok("PlacementModifierManagement is enabled — correct setting.", WHY)
    return flag("PlacementModifierManagement is disabled — it should be enabled for placement bid optimization.", WHY)


def eval_C028(ctx: DatabricksContext) -> ControlResult:
    WHY = "MktStreamHourlyBidAdjustments enables intraday bid changes based on real-time signals. When disabled, the account misses time-of-day optimization."
    sh, df, col, miss = seller_param_row7(ctx, "MktStreamHourlyBidAdjustments", "MktStreamHourlyBidAdjustments|mktstreamhourlybidadjustments|mkt_stream_hourly_bid_adjustments", no_data_flag=False)
    if miss:
        return ControlResult(miss.status, miss.what, WHY)
    b = as_bool(df.iloc[0][col])
    if b is None:
        return ok("MktStreamHourlyBidAdjustments not found in seller params — skipped.", WHY)
    if b is True:
        return ok("MktStreamHourlyBidAdjustments is enabled — correct setting.", WHY)
    return flag("MktStreamHourlyBidAdjustments is disabled — it should be enabled for intraday bid optimization.", WHY)


def eval_C029(ctx: DatabricksContext) -> ControlResult:
    WHY = "AutomaticallyImportCampaigns pulls in campaigns that are not managed by Quartile. When enabled, it can introduce uncontrolled campaigns into the system."
    sh, df, col, miss = seller_param_row7(ctx, "AutomaticallyImportCampaigns", "AutomaticallyImportCampaigns|automaticallyimportcampaigns|automatically_import_campaigns", no_data_flag=False)
    if miss:
        return ControlResult(miss.status, miss.what, WHY)
    b = as_bool(df.iloc[0][col])
    if b is None:
        return ok("AutomaticallyImportCampaigns not found in seller params — skipped.", WHY)
    if b is True:
        return flag("AutomaticallyImportCampaigns is enabled — it should be disabled to prevent uncontrolled campaign imports.", WHY)
    return ok("AutomaticallyImportCampaigns is disabled — correct setting.", WHY)


def eval_C030(ctx: DatabricksContext) -> ControlResult:
    WHY = "StopAudienceAutoLink prevents the system from automatically linking audiences to campaigns. When enabled, it blocks audience-based optimizations."
    sh, df, col, miss = seller_param_row7(ctx, "StopAudienceAutoLink", "StopAudienceAutoLink|stopaudienceautolink|stop_audience_auto_link", no_data_flag=True)
    if miss:
        return ControlResult(miss.status, miss.what, WHY)
    b = as_bool(df.iloc[0][col])
    if b is None:
        return flag(note_data_missing(sh or expected_tab_label("40_Seller_Parameter_Insights_Da"), "StopAudienceAutoLink"), WHY)
    if b is True:
        return flag("StopAudienceAutoLink is enabled — it should be disabled to allow audience auto-linking.", WHY)
    return ok("StopAudienceAutoLink is disabled — correct setting.", WHY)


def eval_C031(ctx: DatabricksContext) -> ControlResult:
    WHY = "IsB2bPlacementManagement enables B2B-specific placement bid adjustments. When disabled, the account is missing B2B placement optimization."
    sh, df, col, miss = seller_param_row7(ctx, "IsB2bPlacementManagement", "IsB2bPlacementManagement|isb2bplacementmanagement|is_b2b_placement_management|b2bplacementmanagement", no_data_flag=True)
    if miss:
        return ControlResult(miss.status, miss.what, WHY)
    b = as_bool(df.iloc[0][col])
    if b is None:
        return flag(note_data_missing(sh or expected_tab_label("40_Seller_Parameter_Insights_Da"), "IsB2bPlacementManagement"), WHY)
    if b is True:
        return ok("IsB2bPlacementManagement is enabled — correct setting.", WHY)
    return flag("IsB2bPlacementManagement is disabled — it should be enabled for B2B placement optimization.", WHY)


def eval_C032(ctx: DatabricksContext) -> ControlResult:
    WHY = "HasDisplayPromote enables the system to use display promotion signals. When disabled, the account cannot benefit from display-driven bid boosts."
    sh, df, col, miss = seller_param_row7(ctx, "HasDisplayPromote", "HasDisplayPromote|hasdisplaypromote|has_display_promote", no_data_flag=True)
    if miss:
        return ControlResult(miss.status, miss.what, WHY)
    b = as_bool(df.iloc[0][col])
    if b is None:
        return flag(note_data_missing(sh or expected_tab_label("40_Seller_Parameter_Insights_Da"), "HasDisplayPromote"), WHY)
    if b is True:
        return ok("HasDisplayPromote is enabled — correct setting.", WHY)
    return flag("HasDisplayPromote is disabled — it should be enabled to allow display promotion signals.", WHY)


def eval_C033(ctx: DatabricksContext) -> ControlResult:
    WHY = "ChangeSBV enables the system to adjust Sponsored Brand Video bids. When disabled, SBV campaigns are not being optimized."
    sh, df, col, miss = seller_param_row7(ctx, "ChangeSBV", "ChangeSBV|changesbv|change_sbv", no_data_flag=True)
    if miss:
        return ControlResult(miss.status, miss.what, WHY)
    b = as_bool(df.iloc[0][col])
    if b is None:
        return flag(note_data_missing(sh or expected_tab_label("40_Seller_Parameter_Insights_Da"), "ChangeSBV"), WHY)
    if b is True:
        return ok("ChangeSBV is enabled — correct setting.", WHY)
    return flag("ChangeSBV is disabled — it should be enabled for SBV bid optimization.", WHY)


# ---- C034: RBO Configuration ----
def eval_C034(ctx: DatabricksContext) -> ControlResult:
    WHY = "Rule-Based Optimization (RBO) rules override the system's automated decisions. Active RBO rules need to be intentional and regularly reviewed."
    sh, df = ds(ctx, "RBO_CONFIG", "33_RBO_Configuration_Insights")
    if df is None or df.empty:
        return ok("No RBO configuration found. No active rules to review.", WHY)

    tab = sh or expected_tab_label("33_RBO_Configuration_Insights")

    if df.shape[1] < 6:
        return flag(note_data_missing(tab, "Data_Type/Setting_Value/Rule_Name"), WHY)

    data_type_col = df.columns[0]
    setting_val_col = df.columns[3]
    rule_name_col = df.columns[5]

    tmp = df.copy()
    tmp["_dtype"] = tmp[data_type_col].astype(str).fillna("").str.strip().str.lower()
    tmp["_setting"] = tmp[setting_val_col].astype(str).fillna("").str.strip()
    tmp["_rulename"] = tmp[rule_name_col].astype(str).fillna("").str.strip()

    rules = tmp[(tmp["_dtype"] == "rules") & ((tmp["_rulename"] != "") | (tmp["_setting"] != ""))].copy()
    if rules.empty:
        return ok("No active RBO rules detected.", WHY)

    n = int(len(rules.index))
    examples = rules["_rulename"].replace("", np.nan).dropna().head(3).tolist()
    ex_txt = ", ".join(examples) if examples else "rule names not available"
    return flag(f"{n} active RBO rule(s) detected. Examples: {ex_txt}. These override automated decisions and need to be reviewed.", WHY)


# ---- C035: SPT Coverage ----
def eval_C035(ctx: DatabricksContext) -> ControlResult:
    WHY = "Categories with 30+ ASINs and 5%+ of total sales should have SPT campaigns. Missing coverage means top categories are not being defended at the search term level."
    sh10, df10 = ds(ctx, "CAMPAIGNS_BY_TYPE", "10_Campaigns_Grouped_by_QT_Camp")
    sh18, df18 = ds(ctx, "PERF_BY_CATEGORY", "18_Performance_by_Category")
    if (df10 is None or df10.empty) or (df18 is None or df18.empty):
        return ok("Campaign type or category performance data not available — SPT check skipped.", WHY)

    asin_cnt = find_col(df18, ["asincount", "asin_count", "asin count"])
    if not asin_cnt:
        asin_cnt = get_col_by_letter(df18, "B")
    sales_pct = find_col(df18, ["totalsalespct", "total_sales_pct", "TotalSalesPct"])
    if not sales_pct:
        sales_pct = get_col_by_letter(df18, "J")
    if not asin_cnt or not sales_pct:
        return ok("Required columns not found in category performance data — SPT check skipped.", WHY)

    tmp18 = df18.copy()
    tmp18["_asin_cnt"] = tmp18[asin_cnt].apply(as_int).fillna(0)
    tmp18["_sales_pct"] = tmp18[sales_pct].apply(as_float).fillna(0.0)
    tmp18["_sales_pct_norm"] = tmp18["_sales_pct"].apply(lambda v: _normalize_pct(v) if v is not None else 0.0)

    req = int(((tmp18["_asin_cnt"] >= 30) & (tmp18["_sales_pct_norm"] >= 5.0)).sum())

    subtype = find_col(df10, ["campaignsubtype"])
    campaigns = find_col(df10, ["campaigns"])
    if not subtype or not campaigns:
        return ok("Campaign subtype column not found — SPT check skipped.", WHY)

    rows = df10[df10[subtype].astype(str).str.strip().str.upper() == "SPT"]
    spt = int(as_int(rows[campaigns].dropna().iloc[0]) or 0) if not rows.empty else 0

    if spt == 0:
        return flag(f"No SPT campaigns found. {req} qualifying categories require SPT coverage.", WHY)
    if req > 0 and spt < req:
        return partial(f"{spt} SPT campaign(s) found but {req} are required based on qualifying categories.", WHY)
    return ok(f"{spt} SPT campaign(s) found — meets the {req} required based on qualifying categories.", WHY)


# ---- C036: CatchAll vs WATM ----
def eval_C036(ctx: DatabricksContext) -> ControlResult:
    WHY = "Accounts that meet the criteria (CPC < $0.50, 200+ ASINs, ACoS target < 15%) need a CatchAll or WATM campaign to capture unstructured traffic."
    sh10, df10 = ds(ctx, "CAMPAIGNS_BY_TYPE", "10_Campaigns_Grouped_by_QT_Camp")
    sh08, df08 = ds(ctx, "CAMPAIGN_REPORT", "08_Campaign_Report")
    sh03, df03 = ds(ctx, "YEARLY_KPIS", "03_Yearly_KPIs")
    sh14, df14 = ds(ctx, "CAMPAIGN_PERF", "14_Campaign_Perf")
    sh41, df41 = ds(ctx, "SELLER_PARAMS", "40_Seller_Params")

    if df10 is None or df10.empty or df08 is None or df08.empty:
        return flag(note_data_missing(expected_tab_label("10_Campaigns_Grouped_by_QT_Camp"), "CampaignSubType/CampaignName"), WHY)

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
                return partial("CatchAll is present even though it is not required for this account. WATM is missing.", WHY)
            return ok(f"CatchAll or WATM is present. CatchAll is not required for this account based on current criteria.", WHY)
        return flag("Neither CatchAll nor WATM was found. At least one is expected for accounts that meet the criteria.", WHY)

    if catchall:
        return ok("CatchAll campaign detected — requirement is met.", WHY)
    if watm > 0 and not catchall:
        return partial(f"CatchAll is required for this account but only WATM ({watm} campaign(s)) was detected.", WHY)
    return flag("CatchAll is required for this account but was not found. No WATM campaign was detected either.", WHY)


# ---- C037: ATM Coverage — Top Sellers ----
def eval_C037(ctx: DatabricksContext) -> ControlResult:
    WHY = "Top-selling ASINs (Tier 30, more than 1.5 orders per day) should have ATM spend assigned. Missing ATM on high-velocity products means losing incremental discovery opportunities."
    sh, df = ds(ctx, "CAMPAIGN_PERF", "14_Campaign_Performance_by_Adve")
    if df is None or df.empty:
        return flag(note_data_missing(expected_tab_label("14_Campaign_Performance_by_Adve"), "ASIN/Orders/ATM_Spend/Tier"), WHY)

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
        return flag(note_data_missing(sh or expected_tab_label("14_Campaign_Performance_by_Adve"), "/".join(missing)), WHY)

    days = getattr(ctx, "window_days", None) or 30

    tmp = df.copy()
    tmp["_asin"] = tmp[asin].astype(str).fillna("").str.strip()
    tmp = tmp[tmp["_asin"] != ""].copy()
    if tmp.empty:
        return ok("No ASIN data found for ATM coverage evaluation.", WHY)

    tmp["_orders"] = tmp[orders].apply(as_float).fillna(0.0)
    tmp["_orders_per_day"] = tmp["_orders"] / float(days)
    tmp["_atm"] = tmp[atm_spend].apply(as_float).fillna(0.0)
    tmp["_tier"] = tmp[tier].astype(str).str.strip().str.upper()

    cond_flag = (
        (tmp["_tier"] == "TIER 30") &
        (tmp["_orders_per_day"] > 1.5) &
        (tmp["_atm"] == 0.0)
    )
    if cond_flag.any():
        n = int(cond_flag.sum())
        asins = tmp[cond_flag]["_asin"].head(3).tolist()
        suffix = f" (showing first {len(asins)} of {n})" if n > 3 else ""
        return flag(f"{n} Tier 30 ASIN(s) with more than 1.5 orders/day have no ATM spend{suffix}: {', '.join(asins)}.", WHY)

    return ok("All Tier 30 ASINs with more than 1.5 orders/day have ATM spend assigned.", WHY)


# ---- C038: ATM Catalog Coverage ----
def eval_C038(ctx: DatabricksContext) -> ControlResult:
    WHY = "ASINs with fewer than 1.5 orders per day should not have ATM spend. Spending ATM budget on low-velocity products wastes resources that should go to better-performing ASINs."
    sh, df = ds(ctx, "CAMPAIGN_PERF", "14_Campaign_Performance_by_Adve")
    if df is None or df.empty:
        return ok("No campaign performance data found — ATM catalog check skipped.", WHY)

    asin = find_col(df, ["asin"])
    orders = find_col(df, ["orders", "purchases"])
    atm_spend = find_col(df, ["atm_spend", "atm spend", "atmspend", "atm"])

    if not asin or not orders or not atm_spend:
        return ok("Required columns not found — ATM catalog check skipped.", WHY)

    days = getattr(ctx, "window_days", None) or 30

    tmp = df.copy()
    tmp["_asin"] = tmp[asin].astype(str).fillna("").str.strip()
    tmp = tmp[tmp["_asin"] != ""].copy()
    if tmp.empty:
        return ok("No ASIN data found for ATM catalog evaluation.", WHY)

    total_asins = len(tmp)
    tmp["_orders"] = tmp[orders].apply(as_float).fillna(0.0)
    tmp["_orders_per_day"] = tmp["_orders"] / float(days)
    tmp["_atm"] = tmp[atm_spend].apply(as_float).fillna(0.0)

    cond = (tmp["_orders_per_day"] < 1.5) & (tmp["_atm"] > 0.0)
    affected = int(cond.sum())

    if total_asins == 0 or affected == 0:
        return ok(f"No low-velocity ASINs with active ATM spend detected across {total_asins} ASINs.", WHY)

    pct = (affected / total_asins) * 100.0

    if pct > 20.0:
        return flag(f"{affected} of {total_asins} ASINs ({pct:.1f}%) have fewer than 1.5 orders/day but still have ATM spend active — above the 20% threshold.", WHY)
    if pct >= 10.0:
        return partial(f"{affected} of {total_asins} ASINs ({pct:.1f}%) have fewer than 1.5 orders/day but still have ATM spend active — in the 10%–20% caution range.", WHY)
    return ok(f"{affected} of {total_asins} ASINs ({pct:.1f}%) have fewer than 1.5 orders/day with ATM spend active — within the acceptable range.", WHY)


# ---- C039: Branded vs Non-Branded Mix ----
def eval_C039(ctx: DatabricksContext) -> ControlResult:
    WHY = "Branded spend and sales should represent 5%–20% of the total. Too low means the brand has no defensive presence. Too high means the budget is over-concentrated on already-loyal customers."
    sh, df = ds(ctx, "SEARCH_TERMS_BY_CATEGORY", "12_Search_Terms_by_Category")
    if df is None or df.empty:
        return flag(note_data_missing(expected_tab_label("12_Search_Terms_by_Category"), "KeywordCategory/ad_spend/ad_sales"), WHY)

    cat_col = find_col(df, ["keywordcategory", "category"])
    spend_col = find_col(df, ["ad_spend"]) or get_col_by_letter(df, "B")
    sales_col = find_col(df, ["ad_sales"]) or get_col_by_letter(df, "D")

    if not cat_col or not spend_col or not sales_col:
        return flag(note_data_missing(sh or expected_tab_label("12_Search_Terms_by_Category"), "KeywordCategory/ad_spend/ad_sales"), WHY)

    branded_mask = df[cat_col].astype(str).str.strip().str.lower() == "branded"
    if not branded_mask.any():
        return flag("Branded row is missing from the Search Terms by Category tab.", WHY)

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
            return flag(f"Branded {label} share is {pct:.1f}% — outside the acceptable range of 1%–35%.", WHY)
        if (1.0 <= pct < 5.0) or (20.0 < pct <= 35.0):
            return partial(f"Branded {label} share is {pct:.1f}% — in the caution range. Preferred range is 5%–20%.", WHY)
        return ok(f"Branded {label} share is {pct:.1f}% — within the acceptable 5%–20% range.", WHY)

    spend_result = _classify(spend_pct, "spend")
    sales_result = _classify(sales_pct, "sales")

    results = [r for r in [spend_result, sales_result] if r is not None]
    if not results:
        return ok("Branded spend and sales share could not be calculated — data may be missing.", WHY)

    statuses = [r.status for r in results]
    if cfg.STATUS_FLAG in statuses:
        return next(r for r in results if r.status == cfg.STATUS_FLAG)
    if cfg.STATUS_PARTIAL in statuses:
        return next(r for r in results if r.status == cfg.STATUS_PARTIAL)
    return ok(f"Branded spend share is {spend_pct:.1f}% and branded sales share is {sales_pct:.1f}% — both within the acceptable range.", WHY)


# ---- C040: SB Spend Share ----
def eval_C040(ctx: DatabricksContext) -> ControlResult:
    WHY = "Sponsored Brands spend should represent 1%–25% of total spend. Outside this range it is either absent or over-concentrated."
    sh, df = ds(ctx, "CAMPAIGNS_BY_AMAZON", "09_Campaigns_Grouped_by_Amazon_")
    if df is None or df.empty:
        return flag(note_data_missing(expected_tab_label("09_Campaigns_Grouped_by_Amazon_"), "Campaign_Type/Perc_Spend"), WHY)
    ctype = find_col(df, ["campaign_type", "type"])
    pct = find_col(df, ["perc_spend", "pct_spend", "spend_pct", "spend pct"])
    if not ctype or not pct:
        return flag(note_data_missing(sh or expected_tab_label("09_Campaigns_Grouped_by_Amazon_"), "Campaign_Type/Perc_Spend"), WHY)
    rows = df[df[ctype].astype(str).str.strip().str.lower() == "sponsored brands"]
    if rows.empty:
        return flag("Sponsored Brands row not found in campaign type data.", WHY)
    v = as_float(rows[pct].dropna().iloc[0]) if not rows[pct].dropna().empty else None
    if v is None:
        return flag("Sponsored Brands spend share value is missing.", WHY)
    v = _normalize_pct(v)
    if v < 1.0 or v > 25.0:
        return flag(f"Sponsored Brands spend share is {v:.1f}% — outside the expected 1%–25% range.", WHY)
    return ok(f"Sponsored Brands spend share is {v:.1f}% — within the expected 1%–25% range.", WHY)


# ---- C041: SB + SBV Combined Spend Share ----
def eval_C041(ctx: DatabricksContext) -> ControlResult:
    WHY = "Combined SB and SBV spend above 40% concentrates too much budget in upper-funnel formats, leaving insufficient investment for conversion-focused campaigns."
    sh, df = ds(ctx, "CAMPAIGNS_BY_AMAZON", "09_Campaigns_Grouped_by_Amazon_")
    if df is None or df.empty:
        return ok("No campaign type data found — SB + SBV spend share check skipped.", WHY)

    ctype = find_col(df, ["campaign_type", "type"])
    spend_col = find_col(df, ["spend"]) or get_col_by_letter(df, "H")
    if not ctype or not spend_col:
        return ok("Required columns not found — SB + SBV spend share check skipped.", WHY)

    tmp = df.copy()
    tmp["_type"] = tmp[ctype].astype(str).str.strip().str.lower()
    tmp["_spend"] = tmp[spend_col].apply(as_float).fillna(0.0)

    total_spend = tmp["_spend"].sum()
    if total_spend == 0:
        return ok("Total spend is zero — SB + SBV share check not applicable.", WHY)

    sb_spend = tmp[tmp["_type"] == "sponsored brands"]["_spend"].sum()
    sbv_spend = tmp[tmp["_type"] == "sponsored brand video"]["_spend"].sum()
    combined = sb_spend + sbv_spend
    pct = (combined / total_spend) * 100.0

    if pct > 40.0:
        return flag(f"Combined SB + SBV spend share is {pct:.1f}% — above the 40% threshold.", WHY)
    if pct > 30.0:
        return partial(f"Combined SB + SBV spend share is {pct:.1f}% — in the 30%–40% caution range.", WHY)
    return ok(f"Combined SB + SBV spend share is {pct:.1f}% — within the acceptable range.", WHY)


# ---- C042: SBV Spend Share Cap ----
def _eval_optional_spend_share_cap(ctx: DatabricksContext, row_label: str, cap: float, why: str) -> ControlResult:
    sh, df = ds(ctx, "CAMPAIGNS_BY_AMAZON", "09_Campaigns_Grouped_by_Amazon_")
    if df is None or df.empty:
        return ok(f"No campaign type data found — {row_label} spend share check skipped.", why)
    ctype = find_col(df, ["campaign_type", "type"])
    pct = find_col(df, ["perc_spend", "pct_spend", "spend_pct", "spend pct"])
    if not ctype or not pct:
        return ok(f"Required columns not found — {row_label} spend share check skipped.", why)
    rows = df[df[ctype].astype(str).str.strip().str.lower() == row_label.lower()]
    if rows.empty:
        return ok(f"No {row_label} campaigns found.", why)
    v = as_float(rows[pct].dropna().iloc[0]) if not rows[pct].dropna().empty else None
    if v is None:
        return ok(f"{row_label} spend share value is missing.", why)
    v = _normalize_pct(v)
    if v > cap:
        return flag(f"{row_label} spend share is {v:.1f}% — above the {cap:.0f}% cap.", why)
    return ok(f"{row_label} spend share is {v:.1f}% — within the {cap:.0f}% cap.", why)


def eval_C042(ctx: DatabricksContext) -> ControlResult:
    WHY = "Sponsored Brand Video spend above 20% of total spend is too concentrated in a single upper-funnel format."
    return _eval_optional_spend_share_cap(ctx, "Sponsored Brand Video", cap=20.0, why=WHY)


# ---- C043: SBTV Spend Share Cap ----
def eval_C043(ctx: DatabricksContext) -> ControlResult:
    WHY = "Sponsored Brand TV Video spend above 10% of total spend is too high for most accounts and can strain overall budget efficiency."
    return _eval_optional_spend_share_cap(ctx, "Sponsored Brand TV Video", cap=10.0, why=WHY)


# ---- C044: SD Defensive Coverage ----
def eval_C044(ctx: DatabricksContext) -> ControlResult:
    WHY = "Categories with 30+ ASINs and 5%+ of total sales should have SD_SPT campaigns for defensive coverage. Missing coverage leaves top categories exposed to competitor targeting."
    sh10, df10 = ds(ctx, "CAMPAIGNS_BY_TYPE", "10_Campaigns_Grouped_by_QT_Camp")
    sh18, df18 = ds(ctx, "PERF_BY_CATEGORY", "18_Performance_by_Category")
    if (df10 is None or df10.empty) or (df18 is None or df18.empty):
        return ok("Campaign type or category performance data not available — SD defensive coverage check skipped.", WHY)

    asin_cnt = find_col(df18, ["asincount", "asin_count", "asin count"]) or get_col_by_letter(df18, "B")
    sales_pct = find_col(df18, ["totalsalespct", "total_sales_pct", "TotalSalesPct"]) or get_col_by_letter(df18, "J")
    if not asin_cnt or not sales_pct:
        return ok("Required columns not found in category performance data — SD defensive coverage check skipped.", WHY)

    tmp18 = df18.copy()
    tmp18["_asin_cnt"] = tmp18[asin_cnt].apply(as_int).fillna(0)
    tmp18["_sales_pct"] = tmp18[sales_pct].apply(as_float).fillna(0.0)
    tmp18["_sales_pct_norm"] = tmp18["_sales_pct"].apply(lambda v: _normalize_pct(v) if v is not None else 0.0)

    req = int(((tmp18["_asin_cnt"] >= 30) & (tmp18["_sales_pct_norm"] >= 5.0)).sum())

    subtype = find_col(df10, ["campaignsubtype"])
    campaigns = find_col(df10, ["campaigns"])
    if not subtype or not campaigns:
        return ok("Campaign subtype column not found — SD defensive coverage check skipped.", WHY)
    rows = df10[df10[subtype].astype(str).str.strip().str.upper() == "SD_SPT"]
    sd_spt = int(as_int(rows[campaigns].dropna().iloc[0]) or 0) if not rows.empty else 0

    if sd_spt == 0:
        return flag(f"No SD_SPT campaigns found. {req} qualifying categories require SD defensive coverage.", WHY)
    if req > 0 and sd_spt < req:
        return partial(f"{sd_spt} SD_SPT campaign(s) found but {req} are required based on qualifying categories.", WHY)
    return ok(f"{sd_spt} SD_SPT campaign(s) found — meets the {req} required based on qualifying categories.", WHY)


# ---- C045: SD VCPM Spend Share ----
def eval_C045(ctx: DatabricksContext) -> ControlResult:
    WHY = "VCPM spend above 10% of total SD spend shifts budget toward impression-based buying that is harder to attribute and optimize."
    sh, df = ds(ctx, "SEARCH_TERMS_BY_CATEGORY", "12_Search_Terms_by_Category")
    if df is None or df.empty:
        return ok("No search terms data found — VCPM spend share check skipped.", WHY)
    cat = find_col(df, ["keywordcategory", "category"])
    spend_pct = find_col(df, ["spend_pct", "spend pct", "pct_spend", "perc_spend"])
    if not cat or not spend_pct:
        return ok("Required columns not found — VCPM spend share check skipped.", WHY)
    rows = df[df[cat].astype(str).str.strip().str.lower().str.contains("vcpm", na=False)]
    if rows.empty:
        return ok("No VCPM rows found in search terms data.", WHY)
    v = as_float(rows[spend_pct].dropna().iloc[0]) if not rows[spend_pct].dropna().empty else None
    if v is None:
        return ok("VCPM spend share value is missing.", WHY)
    v = _normalize_pct(v)
    if v > 10.0:
        return flag(f"VCPM spend share is {v:.1f}% — above the 10% threshold.", WHY)
    if v > 5.0:
        return partial(f"VCPM spend share is {v:.1f}% — in the 5%–10% caution range.", WHY)
    return ok(f"VCPM spend share is {v:.1f}% — within the acceptable range.", WHY)


# ---- C046: SD VCPM Sales Share ----
def eval_C046(ctx: DatabricksContext) -> ControlResult:
    WHY = "VCPM-attributed sales above 20% of total SD sales can misrepresent true performance since VCPM attribution is based on impressions, not clicks."
    sh, df = ds(ctx, "SEARCH_TERMS_BY_CATEGORY", "12_Search_Terms_by_Category")
    if df is None or df.empty:
        return ok("No search terms data found — VCPM sales share check skipped.", WHY)
    cat = find_col(df, ["keywordcategory", "category"])
    sales = find_col(df, ["ad_sales", "adsales", "sales"])
    vcpm_sales = find_col(df, ["vcpm_sales", "vcpm sales"])
    if not cat or not sales:
        return ok("Required columns not found — VCPM sales share check skipped.", WHY)

    if vcpm_sales:
        rows = df[df[cat].astype(str).str.strip().str.lower().str.contains("vcpm", na=False)]
        if rows.empty:
            return ok("No VCPM rows found in search terms data.", WHY)
        v = as_float(rows[vcpm_sales].dropna().iloc[0]) if not rows[vcpm_sales].dropna().empty else None
        total = as_float(df[sales].dropna().sum())
    else:
        rows = df[df[cat].astype(str).str.strip().str.lower().str.contains("vcpm", na=False)]
        if rows.empty:
            return ok("No VCPM rows found in search terms data.", WHY)
        v = as_float(rows[sales].dropna().iloc[0]) if not rows[sales].dropna().empty else None
        total = as_float(df[sales].dropna().sum())

    if v is None or not total or total == 0:
        return ok("VCPM sales share could not be calculated — data may be missing.", WHY)

    share = (v / total) * 100.0
    if share > 20.0:
        return flag(f"VCPM sales share is {share:.1f}% — above the 20% threshold.", WHY)
    return ok(f"VCPM sales share is {share:.1f}% — within the acceptable range.", WHY)


# ---- C047: Out of Budget ----
def eval_C047(ctx: DatabricksContext) -> ControlResult:
    WHY = "Running out of budget means the account stops showing ads for part of the day. Three or more days out of budget in 30 days is a signal that budgets need to be reviewed."
    sh, df = ds(ctx, "ACCOUNT_OOB", "36_Account_Out_of_Budget")
    if df is None or df.empty:
        return ok("No out-of-budget records found.", WHY)

    dt = find_col(df, ["reportdate", "date", "day"]) or get_col_by_letter(df, "A")
    if not dt:
        return ok("Date column not found in out-of-budget data.", WHY)

    if not ctx.ref_date:
        return ok("Reference date not available — out-of-budget check skipped.", WHY)

    all_dts = pd.to_datetime(df[dt], errors="coerce").dropna().dt.normalize()
    if all_dts.empty:
        return ok("No valid dates found in out-of-budget data.", WHY)

    cutoff = pd.Timestamp(ctx.ref_date - timedelta(days=30))
    recent = all_dts[all_dts >= cutoff]
    uniq = len(set(recent.tolist()))

    if uniq >= 3:
        return flag(f"The account was out of budget on {uniq} distinct days in the last 30 days — at or above the 3-day threshold.", WHY)
    if uniq == 2:
        return partial(f"The account was out of budget on 2 distinct days in the last 30 days — approaching the 3-day threshold.", WHY)
    if uniq == 1:
        return ok(f"The account was out of budget on 1 day in the last 30 days — within the acceptable range.", WHY)
    return ok("No out-of-budget days detected in the last 30 days.", WHY)


# ---- C048: Cleanup & Relaunch Cadence (Manual) ----
def eval_C048(ctx: DatabricksContext) -> ControlResult:
    WHY = "Cleanup and relaunch cadence cannot be checked automatically — it requires a manual review of campaign history and structure."
    return ok("Manual check required — this control is reviewed during the QR call.", WHY)


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
    "C001": eval_C001, "C002": eval_C002, "C003": eval_C003,
    "C004": eval_C004, "C005": eval_C005, "C006": eval_C006,
    "C007": eval_C007, "C008": eval_C008, "C009": eval_C009,
    "C010": eval_C010, "C011": eval_C011, "C012": eval_C012,
    "C013": eval_C013, "C014": eval_C014, "C015": eval_C015,
    "C016": eval_C016, "C017": eval_C017, "C018": eval_C018,
    "C019": eval_C019, "C020": eval_C020, "C021": eval_C021,
    "C022": eval_C022, "C023": eval_C023, "C024": eval_C024,
    "C025": eval_C025, "C026": eval_C026, "C027": eval_C027,
    "C028": eval_C028, "C029": eval_C029, "C030": eval_C030,
    "C031": eval_C031, "C032": eval_C032, "C033": eval_C033,
    "C034": eval_C034, "C035": eval_C035, "C036": eval_C036,
    "C037": eval_C037, "C038": eval_C038, "C039": eval_C039,
    "C040": eval_C040, "C041": eval_C041, "C042": eval_C042,
    "C043": eval_C043, "C044": eval_C044, "C045": eval_C045,
    "C046": eval_C046, "C047": eval_C047, "C048": eval_C048,
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
            results[cid] = ControlResult(cfg.STATUS_FLAG, f"Error running {cid}: {type(e).__name__}: {e}", "")
    return results
