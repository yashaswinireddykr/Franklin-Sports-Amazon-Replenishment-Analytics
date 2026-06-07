"""
03_eda.py
---------
Franklin Sports | Amazon Replenishment Analytics

Purpose:
    Exploratory Data Analysis on the merged ASIN-week panel.
    Produces key visualizations saved to the visuals/ directory.

    Analyses covered:
        1.  Dataset structure and schema overview
        2.  Missing value assessment
        3.  Inventory distribution (on-hand units per SKU)
        4.  Forecast distribution (mean vs P90)
        5.  Lead time distribution
        6.  Weeks of Cover (WoC) — full and capped
        7.  Top 15 products by average on-hand inventory
        8.  Inventory vs forecast scatter (alignment check)
        9.  Stockout risk by division (WoC < lead time)
        10. Forecast skew — ASINs where mean > P70

Inputs:  data/clean/merged_final.xlsx
Outputs: visuals/eda_*.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

BASE_DIR    = Path(__file__).resolve().parent.parent
CLEAN_DIR   = BASE_DIR / "data" / "clean"
VISUALS_DIR = BASE_DIR / "visuals"
VISUALS_DIR.mkdir(parents=True, exist_ok=True)

MERGED_PATH = CLEAN_DIR / "merged_final.xlsx"

sns.set(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 150, "axes.grid": True,
                     "grid.linestyle": "--", "grid.alpha": 0.4})


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------

def save_fig(name: str) -> None:
    path = VISUALS_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


def load_merged(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]

    # Coerce key numerics
    for col in ["onhand_units", "forecast_mean", "forecast_p70",
                "forecast_p80", "forecast_p90", "lead_time_days",
                "po_quantity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["start_date", "start_date_pred", "po_request_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


# ------------------------------------------------------------------
# EDA FUNCTIONS
# ------------------------------------------------------------------

def eda_structure(df: pd.DataFrame) -> None:
    """Print dataset shape, column types, and overall missing rate."""
    print(f"\n  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Overall missing rate: {df.isna().mean().mean():.2%}")
    print(f"\n  Column dtypes:\n{df.dtypes.value_counts().to_string()}")


def plot_missing_values(df: pd.DataFrame) -> None:
    """Bar chart of missing value percentage per column."""
    missing = (
        df.isna().mean()
          .sort_values(ascending=False)
          .head(20)
          .reset_index()
    )
    missing.columns = ["column", "missing_pct"]
    missing["missing_pct_label"] = (missing["missing_pct"] * 100).round(1).astype(str) + "%"

    plt.figure(figsize=(10, 7))
    bars = sns.barplot(
        x="missing_pct", y="column", data=missing, color="#4DB6AC"
    )
    for i, row in missing.iterrows():
        bars.text(row["missing_pct"] + 0.001, i,
                  row["missing_pct_label"], va="center", fontsize=9)

    plt.title("Percentage of Missing Values by Column", fontweight="bold")
    plt.xlabel("% Missing")
    plt.ylabel("Column")
    save_fig("eda_missing_values")


def plot_inventory_distribution(df: pd.DataFrame) -> None:
    """Histogram of on-hand inventory units per SKU (capped at 98th pct)."""
    inv = df["onhand_units"].dropna()
    cap = inv.quantile(0.98)

    plt.figure(figsize=(9, 5))
    sns.histplot(inv[inv <= cap], bins=40, kde=True, color="steelblue")
    plt.title("Distribution of On-hand Inventory Units", fontweight="bold")
    plt.xlabel("On-hand Units")
    plt.ylabel("Frequency")
    save_fig("eda_inventory_distribution")


def plot_forecast_distribution(df: pd.DataFrame) -> None:
    """Boxplot comparing forecast mean vs P90 spread."""
    fc_cols = ["forecast_mean", "forecast_p90"]
    available = [c for c in fc_cols if c in df.columns]
    if not available:
        print("  Skipping forecast distribution — columns not found.")
        return

    melted = df[available].melt(var_name="metric", value_name="units").dropna()
    melted = melted[melted["units"] <= melted["units"].quantile(0.995)]

    plt.figure(figsize=(9, 5))
    sns.boxplot(x="units", y="metric", data=melted,
                palette=["#90CAF9", "#FFCC80"], orient="h")
    plt.title("Forecast Distribution: Mean vs P90", fontweight="bold")
    plt.xlabel("Forecasted Units")
    plt.ylabel("")
    save_fig("eda_forecast_distribution_mean_vs_p90")


def plot_lead_time_distribution(df: pd.DataFrame) -> None:
    """Histogram of lead time in days with mean line."""
    if "lead_time_days" not in df.columns:
        print("  Skipping lead time distribution — column not found.")
        return

    lt = df["lead_time_days"].dropna()
    mean_lt = lt.mean()

    plt.figure(figsize=(9, 5))
    sns.histplot(lt, bins=30, kde=True, color="salmon")
    plt.axvline(mean_lt, color="darkred", linestyle="--",
                label=f"Mean = {mean_lt:.1f} days")
    plt.title("Distribution of Lead Time (Days)", fontweight="bold")
    plt.xlabel("Lead Time (Days)")
    plt.ylabel("Count")
    plt.legend()
    save_fig("eda_lead_time_distribution")


def compute_woc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Weeks of Cover (WoC) = on-hand units / forecast mean.
    Filters to rows with positive forecast and valid inventory.
    """
    woc_df = df.dropna(subset=["onhand_units", "forecast_mean"]).copy()
    woc_df = woc_df[woc_df["forecast_mean"] > 0]
    woc_df["woc"] = woc_df["onhand_units"] / woc_df["forecast_mean"]
    if "lead_time_days" in woc_df.columns:
        woc_df["lead_time_weeks"] = woc_df["lead_time_days"] / 7
    return woc_df


def plot_woc_distribution(woc_df: pd.DataFrame) -> None:
    """Two WoC plots: full distribution and capped at 20 weeks."""
    woc = woc_df["woc"].dropna()

    # Full distribution
    plt.figure(figsize=(9, 5))
    plt.plot(sorted(woc), color="#B565A7")
    plt.fill_between(range(len(woc)), sorted(woc), alpha=0.3, color="#B565A7")
    plt.title("Distribution of Weeks of Cover (WoC)", fontweight="bold")
    plt.xlabel("Weeks of Cover")
    plt.ylabel("Number of ASIN-Weeks")
    save_fig("eda_woc_distribution")

    # Capped at 20 weeks
    woc_capped = woc[woc <= 20]
    plt.figure(figsize=(9, 5))
    sns.histplot(woc_capped, bins=40, kde=True, color="#B565A7")
    plt.title("Distribution of Weeks of Cover (Capped at 20 Weeks)", fontweight="bold")
    plt.xlabel("Weeks of Cover (Capped)")
    plt.ylabel("Number of ASIN-Weeks")
    save_fig("eda_woc_distribution_capped")


def plot_avg_inventory_by_product(df: pd.DataFrame) -> None:
    """Horizontal bar chart of top 15 products by average on-hand inventory."""
    group_col = "product" if "product" in df.columns else "division"

    avg_inv = (
        df.groupby(group_col)["onhand_units"]
          .mean()
          .sort_values(ascending=False)
          .head(15)
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(x=avg_inv.values, y=avg_inv.index, palette="crest")
    plt.title(f"Top 15 {group_col.capitalize()}s by Average On-hand Inventory",
              fontweight="bold")
    plt.xlabel("Average On-hand Units")
    plt.ylabel(group_col.capitalize())
    save_fig("eda_avg_inventory_by_product")


def plot_inventory_vs_forecast_scatter(df: pd.DataFrame) -> None:
    """Scatter plot of inventory vs forecast mean to check alignment."""
    sub = df[["onhand_units", "forecast_mean"]].dropna()
    cap_x = sub["forecast_mean"].quantile(0.99)
    cap_y = sub["onhand_units"].quantile(0.99)
    sub = sub[(sub["forecast_mean"] <= cap_x) & (sub["onhand_units"] <= cap_y)]

    plt.figure(figsize=(8, 6))
    plt.scatter(sub["forecast_mean"], sub["onhand_units"],
                alpha=0.3, s=15, color="steelblue")
    # Trend line
    z = np.polyfit(sub["forecast_mean"], sub["onhand_units"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(sub["forecast_mean"].min(), sub["forecast_mean"].max(), 100)
    plt.plot(x_line, p(x_line), "r-", linewidth=2)

    plt.title("Inventory vs Forecast (Same Week)", fontweight="bold")
    plt.xlabel("forecast_mean")
    plt.ylabel("onhand_units")
    save_fig("eda_inventory_vs_forecast_scatter")


def plot_stockout_risk_by_division(woc_df: pd.DataFrame) -> None:
    """
    Bar chart showing % of ASIN-weeks at stockout risk
    (WoC < average lead time in weeks) per division.
    """
    if "division" not in woc_df.columns or "lead_time_weeks" not in woc_df.columns:
        print("  Skipping stockout risk chart — required columns missing.")
        return

    avg_lt = woc_df["lead_time_weeks"].mean()
    woc_df = woc_df.copy()
    woc_df["at_risk"] = (woc_df["woc"] < avg_lt).astype(int)

    risk = (
        woc_df.groupby("division")["at_risk"]
              .mean()
              .sort_values(ascending=False)
              .head(10) * 100
    )

    plt.figure(figsize=(9, 4))
    sns.barplot(x=risk.values, y=risk.index, color="salmon")
    plt.title("Stockout Risk by Division (Top 10)", fontweight="bold")
    plt.xlabel("% of ASIN-Weeks at Risk (WoC < avg lead time)")
    plt.ylabel("Division")
    save_fig("eda_stockout_risk_by_division")


def plot_forecast_skew(df: pd.DataFrame) -> None:
    """
    Bar chart of top 20 ASINs where forecast mean consistently exceeds P70,
    indicating a skewed forecast distribution.
    """
    if not {"forecast_mean", "forecast_p70"}.issubset(df.columns):
        print("  Skipping forecast skew chart — columns not found.")
        return

    df = df.copy()
    df["mean_gt_p70"] = (
        df["forecast_mean"].notna() &
        df["forecast_p70"].notna() &
        (df["forecast_mean"] > df["forecast_p70"])
    ).astype(int)

    skew_summary = (
        df.groupby("asin")["mean_gt_p70"]
          .agg(fraction="mean", weeks="size")
          .reset_index()
          .sort_values("fraction", ascending=False)
          .head(20)
    )

    plt.figure(figsize=(10, 5))
    plt.bar(skew_summary["asin"].astype(str), skew_summary["fraction"])
    plt.xticks(rotation=60, ha="right", fontsize=7)
    plt.ylabel("Fraction of weeks with mean > p70")
    plt.title("Top 20 Skewed ASINs (mean > P70)", fontweight="bold")
    save_fig("mean_gt_p70_top20")


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Franklin Sports | Exploratory Data Analysis")
    print("=" * 60)

    print("\n[1/3] Loading merged dataset...")
    df = load_merged(MERGED_PATH)
    eda_structure(df)

    print("\n[2/3] Computing derived metrics...")
    woc_df = compute_woc(df)
    print(f"  WoC computed for {len(woc_df):,} valid ASIN-week records")
    print(f"  Mean WoC: {woc_df['woc'].mean():.1f} weeks | "
          f"Median: {woc_df['woc'].median():.1f} weeks")

    print("\n[3/3] Generating visualizations...")
    plot_missing_values(df)
    plot_inventory_distribution(df)
    plot_forecast_distribution(df)
    plot_lead_time_distribution(df)
    plot_woc_distribution(woc_df)
    plot_avg_inventory_by_product(df)
    plot_inventory_vs_forecast_scatter(df)
    plot_stockout_risk_by_division(woc_df)
    plot_forecast_skew(df)

    print(f"\nAll visuals saved to: {VISUALS_DIR}")
    print("\nEDA complete.")


if __name__ == "__main__":
    main()
