# config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

# ---------- Status (locked) ----------
STATUS_OK = "OK"
STATUS_FLAG = "FLAG"
STATUS_PARTIAL = "PARTIAL"


# ---------- ControlResult (locked contract) ----------
@dataclass(frozen=True)
class ControlResult:
    status: str
    note: str = ""  # MUST be empty for OK unless manual control policy requires note


# ---------- Databricks workbook sheet candidates ----------
# Key idea: rules_engine calls get_dataset(ctx, dataset_key) and uses these candidates.
# Keep these aligned with your Databricks export naming.
#
# NOTE (2026-03 change):
# Tab "37_Brand_Analytics_Check" removed from export.
# Therefore, ALL tabs that were numbered 37+ shifted down by 1 in the workbook.
#
# NOTE (2026-03 v2 change):
# Controls renumbered C001–C048 (54 → 48 controls).
# Removed controls: C009/C010 (old BA→BAK slot), C043/C044 (old), C050, C053, C054.
# Stale fallback '40_Seller_Params' removed — only '40_Seller_Parameter_Insights_Da' exists in exports.
TAB_CANDIDATES: Dict[str, List[str]] = {
    "ACOS_CHANGES": ["24_Account_ACoS_Changes_History"],

    # Only one active prefix confirmed across all exports
    "SELLER_PARAMS": ["40_Seller_Parameter_Insights_Da"],

    "PRODUCT_LEVEL_ACOS": ["34_Product_Level_ACoS"],
    "CAMPAIGN_LEVEL_ACOS": ["35_Campaign_Level_ACoS"],
    "TIMEFRAME_BOOST": ["27_Timeframe_Boost"],
    "NEGATIVES": ["29_Negative_Keywords__Global"],

    # Primary + short fallback kept for legacy exports
    "CAMPAIGN_PERF": ["14_Campaign_Performance_by_Adve", "14_Campaign_Perf"],

    "BRAND_COMP_TERMS": ["30_Branded_and_Competitor_Terms"],
    "UNMANAGED_ASIN": ["26_Unmanaged_ASIN"],
    "UNMANAGED_BUDGET": ["28_Unmanaged_Budget"],
    "UNMANAGED_CAMPAIGNS": ["31_Unmanaged_campaigns"],
    "UNMANAGED_CAMPAIGN_BUDGETS": ["32_Unmanaged_Campaigns_Budget_O"],
    "ARIS_MANUAL_RECS": ["41_ARIS__Manual_Recomendation"],
    "PORTFOLIO_INSIGHTS": ["25_Portfolio_Insights_and_Confi"],
    "RBO_CONFIG": ["33_RBO_Configuration_Insights"],
    "CAMPAIGNS_BY_TYPE": ["10_Campaigns_Grouped_by_QT_Camp"],
    "PERF_BY_CATEGORY": ["18_Performance_by_Category"],
    "CAMPAIGN_REPORT": ["08_Campaign_Report"],

    # Full tab name confirmed across all exports
    "YEARLY_KPIS": ["03_Yearly_KPIs_Current_vs_Last_"],

    "SEARCH_TERMS_BY_CATEGORY": ["12_Search_Terms_by_Category"],
    "CAMPAIGNS_BY_AMAZON": ["09_Campaigns_Grouped_by_Amazon_"],
    "ACCOUNT_OOB": ["36_Account_Out_of_Budget"],
}


# ---------- Meta extraction hints ----------
# Optional – helps the reader extract account name/ref_date reliably.
ACCOUNT_NAME_KEYS = ["account", "advertiser", "seller", "brand"]
DOWNLOADED_KEYS = ["downloaded", "download timestamp", "exported", "generated"]
