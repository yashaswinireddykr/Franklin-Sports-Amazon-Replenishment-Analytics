"""
01_data_cleaning.py
-------------------
Franklin Sports | Amazon Replenishment Analytics

Purpose:
    Load and clean all five raw input datasets:
        - Product Taxonomy
        - Inventory (weekly on-hand units)
        - Predictive Demand (Amazon forecasts)
        - Open Purchase Orders
        - Lead Time

    Each dataset is standardized, validated, and saved to the
    clean/ directory for use in downstream analysis.

Inputs:  data/raw/*.xlsx
Outputs: data/clean/clean_*.xlsx
         data/clean/validation_report.xlsx
"""

import time
import os
import numpy as np
import pandas as pd
from pathlib import Path


# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

BASE_DIR   = Path(__file__).resolve().parent.parent
RAW_DIR    = BASE_DIR / "data" / "raw"
CLEAN_DIR  = BASE_DIR / "data" / "clean"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

ASSIGNED_GROUP = "racket_scientists"

FILES = {
    "taxonomy":  RAW_DIR / "product_taxonomy_rs.xlsx",
    "inventory": RAW_DIR / "inventory.xlsx",
    "forecast":  RAW_DIR / "predictive_demand.xlsx",
    "open_pos":  RAW_DIR / "open_purchase_orders.xlsx",
    "lead_time": RAW_DIR / "lead_time.xlsx",
}

CLEANED = {
    "taxonomy":  CLEAN_DIR / "clean_product_taxonomy.xlsx",
    "inventory": CLEAN_DIR / "clean_inventory.xlsx",
    "forecast":  CLEAN_DIR / "clean_predictive_demand.xlsx",
    "open_pos":  CLEAN_DIR / "clean_open_purchase_orders.xlsx",
    "lead_time": CLEAN_DIR / "clean_lead_time.xlsx",
}


# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------

def standardize_asin(series: pd.Series) -> pd.Series:
    """Normalize ASIN identifiers to uppercase strings with no extra whitespace."""
    return series.astype(str).str.strip().str.upper()


def to_datetime(series: pd.Series) -> pd.Series:
    """Safely convert a column to datetime, coercing invalid entries to NaT."""
    return pd.to_datetime(series, errors="coerce")


def to_numeric(series: pd.Series) -> pd.Series:
    """Safely convert a column to numeric, coercing invalid values to NaN."""
    return pd.to_numeric(series, errors="coerce")


def dedupe_on_keys(
    df: pd.DataFrame, keys: list, keep: str = "last"
) -> tuple:
    """
    Remove duplicate rows based on specific key columns.

    Returns:
        (deduplicated DataFrame, number of rows removed)
    """
    before = len(df)
    df = df.sort_index().drop_duplicates(subset=keys, keep=keep)
    return df, before - len(df)


def weekly_missing_weeks(
    df: pd.DataFrame, asin_col: str, date_col: str, freq_days: int = 7
) -> pd.DataFrame:
    """
    Identify missing weekly periods for each ASIN.

    Returns a summary DataFrame with date range, observed vs expected
    week counts, and a sample of missing dates per ASIN.
    """
    gaps = []
    for asin, group in df.groupby(asin_col, dropna=False):
        group = group.sort_values(date_col)
        if group[date_col].isna().all() or group.empty:
            continue
        start, end = group[date_col].min(), group[date_col].max()
        expected = pd.date_range(start=start, end=end, freq=f"{freq_days}D")
        missing = sorted(set(expected) - set(group[date_col].dropna()))
        if missing:
            gaps.append({
                "asin": asin,
                "start_min": start,
                "start_max": end,
                "observed_weeks": group[date_col].nunique(),
                "expected_weeks": len(expected),
                "missing_count": len(missing),
                "missing_sample": ", ".join(
                    d.strftime("%Y-%m-%d") for d in missing[:10]
                ),
            })
    return pd.DataFrame(gaps)


# ------------------------------------------------------------------
# LOAD RAW DATA
# ------------------------------------------------------------------

def load_raw_files(files: dict) -> dict:
    """
    Load all raw Excel files into a dictionary of DataFrames.
    Normalizes column names and standardizes ASIN identifiers on load.
    """
    dfs = {}
    for key, path in files.items():
        if not path.exists():
            print(f"  [MISSING] {key}: {path.name}")
            continue
        t0 = time.time()
        df = pd.read_excel(path, dtype=str)
        df.columns = [c.strip() for c in df.columns]
        if "asin" in {c.lower() for c in df.columns}:
            real = [c for c in df.columns if c.lower() == "asin"][0]
            if real != "asin":
                df = df.rename(columns={real: "asin"})
            df["asin"] = standardize_asin(df["asin"])
        dfs[key] = df
        print(f"  Loaded {key:12s} | {len(df):6,} rows | {time.time()-t0:.2f}s")
    return dfs


# ------------------------------------------------------------------
# CLEANING FUNCTIONS
# ------------------------------------------------------------------

def clean_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean product taxonomy:
        - Filter to Baseball division only
        - Exclude Youth/Open product lines
        - Deduplicate on ASIN
    """
    df = df.copy()

    if "division" in df.columns:
        df = df[df["division"].astype(str).str.strip().str.lower() == "baseball"]

    text_cols = [c for c in ["product", "taxonomy", "asin_description"] if c in df.columns]
    if text_cols:
        excl = r"\b(youth|open)\b"
        mask = False
        for c in text_cols:
            mask = mask | df[c].astype(str).str.contains(excl, case=False, na=False)
        df = df.loc[~mask].copy()

    if "asin" in df.columns:
        df, _ = dedupe_on_keys(df, keys=["asin"], keep="first")

    return df.reset_index(drop=True)


def clean_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean inventory data:
        - Parse start_date to datetime
        - Rename quantity column to onhand_units
        - Remove negative or non-numeric quantities
        - Deduplicate on (asin, start_date)
    """
    df = df.copy()

    if "start_date" in df.columns:
        df["start_date"] = to_datetime(df["start_date"])

    src_col = "anon_sellable_onhand_inventory_units_by_asin"
    if src_col in df.columns:
        df = df.rename(columns={src_col: "onhand_units"})

    if "onhand_units" in df.columns:
        df["onhand_units"] = to_numeric(df["onhand_units"])
        df = df[df["onhand_units"].notna() & (df["onhand_units"] >= 0)]

    key_cols = [c for c in ["asin", "start_date"] if c in df.columns]
    if key_cols:
        df, _ = dedupe_on_keys(df, keys=key_cols, keep="last")

    return df.reset_index(drop=True)


def clean_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean predictive demand (forecast) data:
        - Parse start_date to datetime
        - Rename anon_* columns to readable names
        - Coerce forecast values to numeric
        - Deduplicate on (asin, start_date)
    """
    df = df.copy()

    if "start_date" in df.columns:
        df["start_date"] = to_datetime(df["start_date"])

    rename_map = {
        "anon_mean_forecast_units": "forecast_mean",
        "anon_p70_forecast_units":  "forecast_p70",
        "anon_p80_forecast_units":  "forecast_p80",
        "anon_p90_forecast_units":  "forecast_p90",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    for col in ["forecast_mean", "forecast_p70", "forecast_p80", "forecast_p90"]:
        if col in df.columns:
            df[col] = to_numeric(df[col])

    key_cols = [c for c in ["asin", "start_date"] if c in df.columns]
    if key_cols:
        df, _ = dedupe_on_keys(df, keys=key_cols, keep="last")

    return df.reset_index(drop=True)


def clean_open_pos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean open purchase orders:
        - Parse and rename request date to po_request_date
        - Rename quantity column to po_quantity
        - Remove negative or non-numeric quantities
        - Aggregate by (asin, po_request_date, customer_number)
    """
    df = df.copy()

    if "invoiced_or_request_date" in df.columns:
        df = df.rename(columns={"invoiced_or_request_date": "po_request_date"})
    if "po_request_date" in df.columns:
        df["po_request_date"] = to_datetime(df["po_request_date"])

    if "anon_po_quantity" in df.columns:
        df = df.rename(columns={"anon_po_quantity": "po_quantity"})

    if "po_quantity" in df.columns:
        df["po_quantity"] = to_numeric(df["po_quantity"])
        df = df[df["po_quantity"].notna() & (df["po_quantity"] >= 0)]

    if "order_status" in df.columns:
        df = df.drop(columns=["order_status"])

    if "customer_number" in df.columns:
        df["customer_number"] = df["customer_number"].astype(str).str.strip()

    key_cols = {"asin", "po_request_date", "customer_number"}
    if key_cols.issubset(df.columns):
        df = (
            df.groupby(
                ["asin", "po_request_date", "customer_number"], as_index=False
            )["po_quantity"]
            .sum(min_count=1)
        )

    return df.reset_index(drop=True)


def clean_lead_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean lead time data:
        - Coerce lead_time_days to numeric
        - Filter to a valid operational range (7–120 days)
    """
    df = df.copy()

    if "lead_time_days" in df.columns:
        df["lead_time_days"] = to_numeric(df["lead_time_days"])
        df = df[df["lead_time_days"].between(7, 120, inclusive="both")]

    return df.reset_index(drop=True)


# ------------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------------

def run_validation(clean_dir: Path) -> pd.DataFrame:
    """
    Run essential quality checks on all cleaned datasets.
    Returns a summary DataFrame and saves a validation_report.xlsx.
    """
    paths = {
        "taxonomy":  clean_dir / "clean_product_taxonomy.xlsx",
        "inventory": clean_dir / "clean_inventory.xlsx",
        "forecast":  clean_dir / "clean_predictive_demand.xlsx",
        "open_pos":  clean_dir / "clean_open_purchase_orders.xlsx",
        "lead_time": clean_dir / "clean_lead_time.xlsx",
    }

    dfs = {k: pd.read_excel(p) for k, p in paths.items() if p.exists()}
    summary = []

    def add(ds, check, result):
        summary.append({"dataset": ds, "check": check, "result": result})

    tax = dfs.get("taxonomy", pd.DataFrame())
    if not tax.empty and "asin" in tax.columns:
        add("taxonomy", "unique_asins", tax["asin"].nunique())

    inv = dfs.get("inventory", pd.DataFrame())
    if not inv.empty:
        inv["start_date"] = pd.to_datetime(inv.get("start_date"), errors="coerce")
        inv["onhand_units"] = pd.to_numeric(inv.get("onhand_units"), errors="coerce")
        add("inventory", "(asin,start_date) duplicates",
            int(inv.duplicated(subset=["asin", "start_date"]).sum()))
        add("inventory", "negative onhand_units",
            int((inv["onhand_units"] < 0).sum()))

    fc = dfs.get("forecast", pd.DataFrame())
    if not fc.empty:
        for col in ["forecast_p70", "forecast_p80", "forecast_p90"]:
            if col in fc.columns:
                fc[col] = pd.to_numeric(fc[col], errors="coerce")
        if {"forecast_p70", "forecast_p80", "forecast_p90"}.issubset(fc.columns):
            add("forecast", "p70>p80 violations",
                int((fc["forecast_p70"] > fc["forecast_p80"]).sum()))
            add("forecast", "p80>p90 violations",
                int((fc["forecast_p80"] > fc["forecast_p90"]).sum()))

    lt = dfs.get("lead_time", pd.DataFrame())
    if not lt.empty and "lead_time_days" in lt.columns:
        lt["lead_time_days"] = pd.to_numeric(lt["lead_time_days"], errors="coerce")
        add("lead_time", "out_of_range (7-120 days)",
            int(((lt["lead_time_days"] < 7) | (lt["lead_time_days"] > 120)).sum()))

    summary_df = pd.DataFrame(summary)
    report_path = clean_dir / "validation_report.xlsx"
    summary_df.to_excel(report_path, index=False)
    print(f"\nValidation report saved: {report_path.name}")
    return summary_df


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Franklin Sports | Data Cleaning Pipeline")
    print("=" * 60)

    print("\n[1/3] Loading raw files...")
    dfs = load_raw_files(FILES)

    print("\n[2/3] Cleaning datasets...")
    cleaners = {
        "taxonomy":  clean_taxonomy,
        "inventory": clean_inventory,
        "forecast":  clean_forecast,
        "open_pos":  clean_open_pos,
        "lead_time": clean_lead_time,
    }
    for key, fn in cleaners.items():
        if key not in dfs:
            print(f"  Skipping {key} — not loaded.")
            continue
        cleaned = fn(dfs[key])
        cleaned.to_excel(CLEANED[key], index=False, engine="xlsxwriter")
        print(f"  Saved {key:12s} → {CLEANED[key].name} ({len(cleaned):,} rows)")

    print("\n[3/3] Running validation checks...")
    summary = run_validation(CLEAN_DIR)
    print(summary.to_string(index=False))

    print("\nData cleaning complete.")


if __name__ == "__main__":
    main()
