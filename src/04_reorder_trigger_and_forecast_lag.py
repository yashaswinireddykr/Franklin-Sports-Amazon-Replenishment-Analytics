"""
04_reorder_trigger_and_forecast_lag.py
---------------------------------------
Franklin Sports | Amazon Replenishment Analytics

Purpose:
    Objective 1 — Reorder Trigger (Weeks of Cover)
        Identifies the inventory coverage level at which Amazon
        is most likely to place a purchase order. Derives a
        data-driven reorder trigger and operational reorder zone
        using empirical PO probability curves.

    Objective 2 — Forecast Signal & Lag Analysis
        Quantifies how far in advance Amazon's demand forecasts
        signal upcoming purchase orders. Measures forecast-level
        correlation at lags 1–6 weeks, week-over-week forecast
        change correlation, and surge-to-PO timing.

Key Findings:
    - Amazon reorders consistently when WoC falls between 4–8 weeks,
      with peak probability near 6 WoC.
    - Forecast surges (>20% WoW increase) precede POs by ~4 weeks,
      creating a 6–7 week forward planning window for Franklin.

Inputs:  data/clean/merged_final.xlsx
         data/clean/validation_data.xlsx  (optional, for held-out validation)
Outputs: Console metrics and matplotlib charts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

BASE_DIR  = Path(__file__).resolve().parent.parent
CLEAN_DIR = BASE_DIR / "data" / "clean"

MERGED_PATH = CLEAN_DIR / "merged_final.xlsx"
VAL_PATH    = CLEAN_DIR / "validation_data.xlsx"   # optional

WOC_MIN, WOC_MAX   = 0, 14
MIN_BIN_COUNT      = 200
SURGE_THRESHOLD    = 0.20     # 20% WoW forecast increase = surge
MAX_LAG            = 6        # weeks

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True,
                     "grid.linestyle": "--", "grid.alpha": 0.35})


# ------------------------------------------------------------------
# DATA PREPARATION
# ------------------------------------------------------------------

def load_and_prepare(path: Path) -> pd.DataFrame:
    """
    Load merged dataset, filter to Baseball division, and compute
    core derived fields: week_monday, WoC, and PO flag.
    """
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]

    # Coerce numeric fields
    for col in ["onhand_units", "forecast_mean", "forecast_p70",
                "po_quantity", "lead_time_days"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Date alignment
    date_col = next((c for c in ["start_date", "week_start_date"] if c in df.columns), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["week_monday"] = df[date_col] - pd.to_timedelta(
            df[date_col].dt.weekday, unit="D"
        )

    # Filter to Baseball division
    if "division" in df.columns:
        bb_mask = df["division"].str.lower().eq("baseball")
        if bb_mask.any():
            df = df.loc[bb_mask].copy()

    # Derived fields
    df["woc"] = np.where(
        df["forecast_mean"] > 0,
        df["onhand_units"] / df["forecast_mean"],
        np.nan
    )
    df["po_quantity"] = df["po_quantity"].fillna(0)
    df["po_flag"]     = (df["po_quantity"] > 0).astype(int)

    if "lead_time_days" in df.columns:
        df["lead_weeks"] = df["lead_time_days"] / 7

    return df


# ------------------------------------------------------------------
# OBJECTIVE 1 — REORDER TRIGGER (WEEKS OF COVER)
# ------------------------------------------------------------------

def build_woc_curve(df: pd.DataFrame, po_flag_col: str = "po_flag") -> pd.DataFrame:
    """
    Build an empirical PO probability curve binned by WoC level.
    Applies rolling smoothing for stability.
    """
    valid = df[
        np.isfinite(df["woc"]) &
        (df["woc"] >= WOC_MIN) &
        (df["woc"] <= WOC_MAX)
    ].copy()
    valid["woc_round"] = valid["woc"].round().astype(int)

    curve = (
        valid.groupby("woc_round")
             .agg(po_prob=(po_flag_col, "mean"), count=(po_flag_col, "size"))
             .reset_index()
             .sort_values("woc_round")
    )
    core = curve[curve["count"] >= MIN_BIN_COUNT].copy()
    if core.empty or core["woc_round"].nunique() < 6:
        core = curve.copy()

    core["po_smooth"] = core["po_prob"].rolling(window=3, center=True, min_periods=1).mean()
    return core, valid


def find_reorder_zone(curve: pd.DataFrame, tol_pp: float = 5.0) -> dict:
    """
    Identify the empirical reorder trigger (peak WoC) and the
    operational reorder zone (within ±5pp of the peak).
    """
    in_band = curve[(curve["woc_round"] >= 2) & (curve["woc_round"] <= 8)]
    peak_idx  = in_band["po_smooth"].idxmax() if not in_band.empty else curve["po_smooth"].idxmax()
    peak_woc  = int(curve.loc[peak_idx, "woc_round"])
    peak_prob = float(curve.loc[peak_idx, "po_smooth"])

    threshold  = peak_prob - (tol_pp / 100)
    near_weeks = curve.loc[curve["po_smooth"] >= threshold, "woc_round"]
    zone_start = max(4, int(near_weeks.min()) if not near_weeks.empty else peak_woc - 2)
    zone_end   = min(8, int(near_weeks.max()) if not near_weeks.empty else peak_woc + 2)

    return {"peak_woc": peak_woc, "peak_prob": peak_prob,
            "zone_start": zone_start, "zone_end": zone_end}


def plot_woc_trigger(curve: pd.DataFrame, zone: dict, label: str = "TRAINING") -> None:
    """Plot the smoothed PO probability curve with reorder zone overlay."""
    plt.figure(figsize=(8, 5))
    plt.plot(curve["woc_round"], curve["po_smooth"] * 100,
             marker="o", linewidth=2, label=f"{label} smoothed PO probability")
    plt.axvspan(zone["zone_start"], zone["zone_end"],
                color="orange", alpha=0.15,
                label=f"{label} reorder zone ≈ {zone['zone_start']}–{zone['zone_end']} WoC")
    plt.axvline(zone["peak_woc"], color="gray", linestyle=":",
                label=f"{label} peak ≈ {zone['peak_woc']} WoC")
    plt.title(f"{label} — Weeks of Cover vs Purchase-Order Probability")
    plt.xlabel("Weeks of Cover")
    plt.ylabel("PO Probability (%)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_objective_1(df: pd.DataFrame) -> dict:
    """
    Full Objective 1 pipeline:
        1. Build WoC probability curve
        2. Identify reorder trigger and operational zone
        3. Validate against lead time
        4. Precision-Recall sweep across WoC thresholds
    """
    print("\n" + "=" * 50)
    print("OBJECTIVE 1 — Reorder Trigger (Weeks of Cover)")
    print("=" * 50)

    curve, valid = build_woc_curve(df)
    zone = find_reorder_zone(curve)

    # Evidence shares
    in_zone = valid[
        (valid["woc_round"] >= zone["zone_start"]) &
        (valid["woc_round"] <= zone["zone_end"])
    ]
    rows_share = 100 * len(in_zone) / max(1, len(valid))
    po_share   = 100 * in_zone["po_flag"].sum() / max(1, valid["po_flag"].sum())

    plot_woc_trigger(curve, zone)

    print(f"\n  Trigger (empirical peak) : ≈ {zone['peak_woc']} WoC "
          f"(PO prob ≈ {zone['peak_prob']*100:.1f}%)")
    print(f"  Operational reorder zone : {zone['zone_start']}–{zone['zone_end']} WoC")
    print(f"  Share of rows in zone    : {rows_share:.1f}%")
    print(f"  Share of all POs in zone : {po_share:.1f}%")

    # Lead-time buffer validation
    if "lead_weeks" in df.columns:
        mean_lt = df["lead_weeks"].mean()
        buffer  = zone["peak_woc"] - mean_lt
        print(f"\n  Average lead time        : ≈ {mean_lt:.1f} weeks")
        print(f"  Buffer at {zone['peak_woc']} WoC trigger  : ≈ {buffer:.1f} weeks")

    # Precision-Recall sweep
    rows_pr = []
    for t in np.arange(3, 10.25, 0.25):
        pred = (valid["woc"] <= t).astype(int)
        TP = ((pred == 1) & (valid["po_flag"] == 1)).sum()
        FP = ((pred == 1) & (valid["po_flag"] == 0)).sum()
        FN = ((pred == 0) & (valid["po_flag"] == 1)).sum()
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0
        rec  = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        rows_pr.append({"threshold": t, "precision": prec, "recall": rec, "f1": f1})

    sweep = pd.DataFrame(rows_pr)
    best  = sweep.loc[sweep["f1"].idxmax()]

    plt.figure(figsize=(8, 4))
    plt.plot(sweep["threshold"], sweep["precision"], label="Precision")
    plt.plot(sweep["threshold"], sweep["recall"],    label="Recall")
    plt.plot(sweep["threshold"], sweep["f1"],        label="F1")
    plt.axvline(float(best["threshold"]), color="black", ls="-",
                label=f"Best F1 ≈ {best['threshold']:.1f} WoC")
    plt.title("Precision–Recall–F1 vs WoC Threshold")
    plt.xlabel("Weeks of Cover")
    plt.ylabel("Metric")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"\n  Best F1 threshold : ≈ {best['threshold']:.2f} WoC | "
          f"Precision={best['precision']:.2f} | Recall={best['recall']:.2f}")

    return zone


# ------------------------------------------------------------------
# OBJECTIVE 2 — FORECAST SIGNAL & LAG ANALYSIS
# ------------------------------------------------------------------

def lag_correlation(df: pd.DataFrame, po_flag_col: str = "po_flag",
                    label: str = "TRAINING") -> pd.Series:
    """
    Compute Pearson correlation between lagged forecast_mean values
    and PO occurrence at each lag from 1 to MAX_LAG weeks.
    """
    df2 = df.sort_values(["asin", "week_monday"]).copy()

    for lag in range(1, MAX_LAG + 1):
        df2[f"forecast_lag{lag}"] = df2.groupby("asin")["forecast_mean"].shift(lag)

    corrs = {}
    for lag in range(1, MAX_LAG + 1):
        sub = df2[[po_flag_col, f"forecast_lag{lag}"]].dropna()
        corrs[lag] = sub.corr().iloc[0, 1] if not sub.empty else np.nan

    corr_s = pd.Series(corrs, name=f"{label}_lag_corr")
    best_lag  = int(corr_s.idxmax())
    best_corr = float(corr_s.max())

    plt.figure(figsize=(7, 4))
    plt.plot(corr_s.index, corr_s.values, marker="o", linewidth=2)
    plt.axhline(0, color="gray", linestyle="--", alpha=0.6)
    plt.ylim(corr_s.min() - 0.005, corr_s.max() + 0.005)
    plt.scatter(best_lag, best_corr, color="orange", s=100, zorder=3,
                label=f"Peak ≈ {best_lag} weeks (r={best_corr:.3f})")
    plt.title(f"{label} — Forecast → PO Correlation by Lag (weeks)")
    plt.xlabel("Lag (weeks before PO)")
    plt.ylabel("Correlation with PO occurrence")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"\n  [{label}] Strongest forecast signal: {best_lag} week(s) before PO "
          f"(r = {best_corr:.4f})")
    return corr_s


def delta_forecast_po_timing(df: pd.DataFrame, po_flag_col: str = "po_flag",
                              label: str = "TRAINING") -> pd.Series:
    """
    Measure correlation between week-over-week forecast changes
    and PO occurrence 1–MAX_LAG weeks in the future.
    """
    df2 = df.sort_values(["asin", "week_monday"]).copy()
    df2["forecast_change"] = df2.groupby("asin")["forecast_mean"].diff()

    corrs = {}
    for k in range(1, MAX_LAG + 1):
        df2[f"po_lead{k}"] = df2.groupby("asin")[po_flag_col].shift(-k)
        sub = df2[["forecast_change", f"po_lead{k}"]].dropna()
        corrs[k] = sub["forecast_change"].corr(sub[f"po_lead{k}"]) if not sub.empty else np.nan

    corr_s = pd.Series(corrs, name=f"{label}_delta_corr")
    best_lag  = int(corr_s.idxmax())
    best_corr = float(corr_s.max())

    pad = max(0.001, 0.15 * (corr_s.max() - corr_s.min() + 1e-9))
    plt.figure(figsize=(7, 4))
    plt.plot(corr_s.index, corr_s.values, marker="o", linewidth=2)
    plt.axhline(0, color="gray", ls="--", alpha=0.6)
    plt.ylim(corr_s.min() - pad, corr_s.max() + pad)
    plt.scatter(best_lag, best_corr, color="#ff9800", s=90, zorder=3,
                label=f"Peak ≈ {best_lag} wks (r={best_corr:.4f})")
    plt.title(f"{label} — Forecast Change vs Future Purchase Orders")
    plt.xlabel("Weeks before PO occurs")
    plt.ylabel("Correlation (Δforecast vs future PO)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"  [{label}] Δforecast peak timing: {best_lag} weeks before PO "
          f"(r = {best_corr:.4f})")
    return corr_s


def forecast_surge_timing(df: pd.DataFrame, po_flag_col: str = "po_flag",
                           label: str = "TRAINING") -> pd.DataFrame:
    """
    Flag weeks where forecast jumps >SURGE_THRESHOLD (WoW change),
    then measure PO probability at each lag 1–MAX_LAG weeks later.
    """
    df2 = df.sort_values(["asin", "week_monday"]).copy()
    df2["forecast_change"] = df2.groupby("asin")["forecast_mean"].pct_change()
    df2["surge_flag"] = (df2["forecast_change"] > SURGE_THRESHOLD).astype(int)

    surge_count = df2["surge_flag"].sum()
    print(f"\n  [{label}] Surge events detected: {surge_count:,} "
          f"({df2['surge_flag'].mean()*100:.2f}% of records, "
          f"threshold = {SURGE_THRESHOLD:.0%})")

    po_probs = []
    for lag in range(1, MAX_LAG + 1):
        df2[f"po_future{lag}"] = df2.groupby("asin")[po_flag_col].shift(-lag)
        prob = df2.loc[df2["surge_flag"] == 1, f"po_future{lag}"].mean()
        po_probs.append(prob)

    lag_df = pd.DataFrame({
        "weeks_ahead": range(1, MAX_LAG + 1),
        "po_prob_after_surge": po_probs
    })

    best_lag  = int(lag_df.loc[lag_df["po_prob_after_surge"].idxmax(), "weeks_ahead"])
    best_prob = float(lag_df["po_prob_after_surge"].max())

    plt.figure(figsize=(7, 4))
    plt.plot(lag_df["weeks_ahead"], lag_df["po_prob_after_surge"] * 100,
             marker="o", linewidth=2, color="teal")
    plt.title(f"{label} — Forecast Surges and Subsequent PO Activity")
    plt.xlabel("Weeks after forecast surge")
    plt.ylabel("PO Probability (%)")
    plt.tight_layout()
    plt.show()

    print(f"  [{label}] Peak PO response: {best_lag} week(s) after surge "
          f"(PO prob = {best_prob*100:.1f}%)")
    return lag_df


def run_objective_2(df: pd.DataFrame) -> None:
    """Full Objective 2 pipeline: lag correlation, delta analysis, surge timing."""
    print("\n" + "=" * 50)
    print("OBJECTIVE 2 — Forecast Signal & Lag Analysis")
    print("=" * 50)

    lag_correlation(df, label="TRAINING")
    delta_forecast_po_timing(df, label="TRAINING")
    forecast_surge_timing(df, label="TRAINING")


# ------------------------------------------------------------------
# VALIDATION (TRAINING vs HELD-OUT DATA)
# ------------------------------------------------------------------

def run_validation(df_train: pd.DataFrame, val_path: Path) -> None:
    """
    If a validation dataset is available, replay Obj 1 and 2
    on the held-out data and compare results to training.
    """
    if not val_path.exists():
        print("\n  Validation file not found — skipping held-out comparison.")
        return

    print("\n" + "=" * 50)
    print("VALIDATION — Training vs Held-out Comparison")
    print("=" * 50)

    val = pd.read_excel(val_path)
    val["date"] = pd.to_datetime(val.get("date", val.get("po_request_date")), errors="coerce")
    val["week_monday"] = val["date"] - pd.to_timedelta(val["date"].dt.weekday, unit="D")

    val_weekly = (
        val.groupby(["asin", "week_monday"], as_index=False)["quantity"]
           .sum()
           .rename(columns={"quantity": "po_quantity_val"})
    )

    val_panel = df_train.merge(val_weekly, on=["asin", "week_monday"], how="left")
    val_panel["po_quantity_val"] = val_panel["po_quantity_val"].fillna(0)
    val_panel["po_flag_val"]     = (val_panel["po_quantity_val"] > 0).astype(int)

    # Objective 1 validation
    curve_val, valid_val = build_woc_curve(val_panel, po_flag_col="po_flag_val")
    zone_val = find_reorder_zone(curve_val)
    plot_woc_trigger(curve_val, zone_val, label="VALIDATION")

    print(f"\n  [VALIDATION] Trigger: ≈ {zone_val['peak_woc']} WoC | "
          f"Zone: {zone_val['zone_start']}–{zone_val['zone_end']} WoC")

    # Objective 2 validation
    lag_correlation(val_panel, po_flag_col="po_flag_val", label="VALIDATION")
    forecast_surge_timing(val_panel, po_flag_col="po_flag_val", label="VALIDATION")


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Franklin Sports | Reorder Trigger & Forecast Lag Analysis")
    print("=" * 60)

    print("\nLoading data...")
    df = load_and_prepare(MERGED_PATH)
    print(f"  Baseball division rows: {len(df):,}")

    zone = run_objective_1(df)
    run_objective_2(df)
    run_validation(df, VAL_PATH)

    print("\nObjectives 1 & 2 complete.")


if __name__ == "__main__":
    main()
