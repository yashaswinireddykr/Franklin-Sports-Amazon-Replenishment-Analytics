"""
02_merge_datasets.py
--------------------
Franklin Sports | Amazon Replenishment Analytics

Purpose:
    Merge all five cleaned datasets into a single ASIN-week level
    analytical panel used across all four modeling objectives.

    Merge logic:
        1. Taxonomy is the master key space (Baseball division only)
        2. Lead time is joined into purchase orders on customer_number
        3. Inventory, forecast, and PO+lead_time are left-joined onto
           taxonomy by ASIN

    Post-merge integrity checks confirm ASIN coverage and cardinality.

Inputs:  data/clean/clean_*.xlsx
Outputs: data/clean/merged_final.xlsx
"""

import pandas as pd
from pathlib import Path


# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

BASE_DIR  = Path(__file__).resolve().parent.parent
CLEAN_DIR = BASE_DIR / "data" / "clean"

OUTPUT_PATH = CLEAN_DIR / "merged_final.xlsx"


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------

def normalize_asin(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ASIN column is uppercase string with no whitespace."""
    if "asin" in df.columns:
        df["asin"] = df["asin"].astype(str).str.strip().str.upper()
    return df


def normalize_customer_number(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure customer_number is a consistent string key."""
    if "customer_number" in df.columns:
        df["customer_number"] = df["customer_number"].astype(str).str.strip()
    return df


# ------------------------------------------------------------------
# MERGE PIPELINE
# ------------------------------------------------------------------

def load_cleaned_datasets(clean_dir: Path) -> dict:
    """Load all five cleaned datasets from the clean directory."""
    paths = {
        "taxonomy":   clean_dir / "clean_product_taxonomy.xlsx",
        "inventory":  clean_dir / "clean_inventory.xlsx",
        "forecast":   clean_dir / "clean_predictive_demand.xlsx",
        "open_pos":   clean_dir / "clean_open_purchase_orders.xlsx",
        "lead_time":  clean_dir / "clean_lead_time.xlsx",
    }
    dfs = {}
    for key, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Expected cleaned file not found: {path}")
        df = pd.read_excel(path)
        df.columns = [c.strip() for c in df.columns]
        dfs[key] = df
        print(f"  Loaded {key:12s} → {len(df):6,} rows")
    return dfs


def merge_datasets(dfs: dict) -> pd.DataFrame:
    """
    Execute the full merge sequence:
        1. Normalize keys
        2. Join lead_time into open_pos on customer_number
        3. Left-join all datasets onto taxonomy by ASIN
    """
    taxonomy  = normalize_asin(dfs["taxonomy"].copy())
    inventory = normalize_asin(dfs["inventory"].copy())
    forecast  = normalize_asin(dfs["forecast"].copy())
    open_pos  = normalize_asin(normalize_customer_number(dfs["open_pos"].copy()))
    lead_time = normalize_customer_number(dfs["lead_time"].copy())

    # Step 1: Attach lead time to purchase orders
    po_with_lt = open_pos.merge(
        lead_time, on="customer_number", how="left", suffixes=("", "_lt")
    )

    # Step 2: Build merged panel from taxonomy outward
    merged = taxonomy.copy()
    merged = merged.merge(inventory,  on="asin", how="left", suffixes=("", "_inv"))
    merged = merged.merge(forecast,   on="asin", how="left", suffixes=("", "_pred"))
    merged = merged.merge(po_with_lt, on="asin", how="left", suffixes=("", "_po"))

    return merged


def validate_merge(merged: pd.DataFrame, taxonomy: pd.DataFrame) -> None:
    """
    Print post-merge integrity checks:
        - ASIN coverage (all taxonomy ASINs present)
        - Unexpected ASINs not in taxonomy
        - Row cardinality per ASIN
    """
    merged_asins   = set(merged["asin"].astype(str).str.strip().str.upper())
    taxonomy_asins = set(taxonomy["asin"].astype(str).str.strip().str.upper())

    extra   = merged_asins - taxonomy_asins
    missing = taxonomy_asins - merged_asins

    print(f"\n  Taxonomy ASINs  : {len(taxonomy_asins):,}")
    print(f"  Merged ASINs    : {len(merged_asins):,}")
    print(f"  Extra (not in taxonomy): {len(extra)}")
    print(f"  Missing from merged    : {len(missing)}")

    if not missing:
        print("  ✓ All taxonomy ASINs present in merged output.")

    rows_per_asin = merged.groupby("asin").size().describe()
    print(f"\n  Rows per ASIN — mean: {rows_per_asin['mean']:.1f} "
          f"| max: {rows_per_asin['max']:.0f} "
          f"| min: {rows_per_asin['min']:.0f}")


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Franklin Sports | Dataset Merge Pipeline")
    print("=" * 60)

    print("\n[1/3] Loading cleaned datasets...")
    dfs = load_cleaned_datasets(CLEAN_DIR)

    print("\n[2/3] Merging datasets...")
    merged = merge_datasets(dfs)
    print(f"\n  Final merged shape: {merged.shape[0]:,} rows × {merged.shape[1]} columns")

    print("\n[3/3] Running integrity checks...")
    validate_merge(merged, dfs["taxonomy"])

    merged.to_excel(OUTPUT_PATH, index=False, engine="xlsxwriter")
    print(f"\n  Saved: {OUTPUT_PATH.name}")
    print("\nMerge pipeline complete.")


if __name__ == "__main__":
    main()
