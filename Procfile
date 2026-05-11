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
    what: str = ""   # What We Saw — always populated, plain language + numbers
    why: str = ""    # Why It Matters — one short action-oriented sentence


# ---------- Databricks workbook sheet candidates ----------
TAB_CANDIDATES: Dict[str, List[str]] = {
    "ACOS_CHANGES": ["24_Account_ACoS_Changes_History"],
    "SELLER_PARAMS": ["40_Seller_Parameter_Insights_Da"],
    "PRODUCT_LEVEL_ACOS": ["34_Product_Level_ACoS"],
    "CAMPAIGN_LEVEL_ACOS": ["35_Campaign_Level_ACoS"],
    "TIMEFRAME_BOOST": ["27_Timeframe_Boost"],
    "NEGATIVES": ["29_Negative_Keywords__Global"],
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
    "YEARLY_KPIS": ["03_Yearly_KPIs_Current_vs_Last_"],
    "SEARCH_TERMS_BY_CATEGORY": ["12_Search_Terms_by_Category"],
    "CAMPAIGNS_BY_AMAZON": ["09_Campaigns_Grouped_by_Amazon_"],
    "ACCOUNT_OOB": ["36_Account_Out_of_Budget"],
}


# ---------- Meta extraction hints ----------
ACCOUNT_NAME_KEYS = ["account", "advertiser", "seller", "brand"]
DOWNLOADED_KEYS = ["downloaded", "download timestamp", "exported", "generated"]
