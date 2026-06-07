"""
05_po_forecasting.py
--------------------
Franklin Sports | Amazon Replenishment Analytics

Purpose:
    Objective 3 — Weekly Purchase Order Forecasting

    Predicts when Amazon will place a purchase order for a specific
    ASIN (occurrence) and how many units it will order (quantity),
    using demand forecasts, Weeks of Cover, and seasonal signals.

    Models:
        - Linear Regression   : PO quantity prediction
        - Logistic Regression : PO occurrence (will Amazon order?)
        - Random Forest       : PO occurrence (stronger pattern detection)

    Key Results:
        - Random Forest Precision ≈ 0.83 | Recall ≈ 0.74 | AUC ≈ 0.80
        - WoC and lagged forecast shifts are the strongest PO predictors
        - Holiday flags materially improve prediction of seasonal spikes

Inputs:  data/clean/merged_final.xlsx
         data/clean/validation_data.xlsx  (optional)
Outputs: Trained model objects (in-memory), forecast tables printed to console
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model  import LinearRegression, LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from sklearn.metrics       import (
    mean_absolute_error, mean_absolute_percentage_error,
    precision_score, recall_score, f1_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

BASE_DIR  = Path(__file__).resolve().parent.parent
CLEAN_DIR = BASE_DIR / "data" / "clean"

MERGED_PATH = CLEAN_DIR / "merged_final.xlsx"
VAL_PATH    = CLEAN_DIR / "validation_data.xlsx"

OCC_THRESHOLD = 0.35   # PO occurrence probability threshold


# ------------------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------------------

def build_holiday_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary event flags for key seasonal ordering periods:
    Halloween, Thanksgiving, Cyber Monday, Christmas.
    Amazon's ordering behavior shifts materially around these dates.
    """
    df = df.copy()
    year = int(df["week_monday"].dt.year.min())

    halloween = pd.to_datetime(f"{year}-10-31")
    nov_days  = pd.date_range(f"{year}-11-01", f"{year}-11-30", freq="D")
    thanksgiving = nov_days[nov_days.weekday == 3][3]
    cyber_monday = thanksgiving + pd.Timedelta(days=4)
    christmas    = pd.to_datetime(f"{year}-12-25")

    def to_monday(d):
        return d - pd.to_timedelta(d.weekday(), unit="D")

    hw_mon = to_monday(halloween)
    th_mon = to_monday(thanksgiving)
    cy_mon = to_monday(cyber_monday)
    xm_mon = to_monday(christmas)

    df["evt_halloween"] = (df["week_monday"] == hw_mon).astype(int)
    df["evt_holiday"]   = (df["week_monday"] == th_mon).astype(int)
    df["evt_cyber"]     = (df["week_monday"] == cy_mon).astype(int)
    df["evt_christmas"] = (df["week_monday"] == xm_mon).astype(int)

    return df


def build_panel(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the modeling panel from the merged dataset:
        - Filter to Baseball division
        - Align dates to Monday-based week index
        - Compute WoC and lagged forecast features
        - Add seasonal holiday flags
    """
    df = df_raw.copy()
    df.columns = [c.strip() for c in df.columns]

    # Coerce numerics
    for col in ["onhand_units", "forecast_mean", "po_quantity", "lead_time_days"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Date alignment
    date_col = next((c for c in ["start_date", "week_monday"] if c in df.columns), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["week_monday"] = df[date_col] - pd.to_timedelta(
            df[date_col].dt.weekday, unit="D"
        )

    # Baseball-only filter
    if "division" in df.columns:
        bb = df["division"].str.lower().eq("baseball")
        if bb.any():
            df = df.loc[bb].copy()

    # Core derived fields
    df["forecast_mean"] = df["forecast_mean"].fillna(0)
    df["po_quantity"]   = df.get("po_quantity", pd.Series(np.nan, index=df.index)).fillna(0)

    onhand = df.get("onhand_units", pd.Series(np.nan, index=df.index))
    df["woc"] = (onhand / df["forecast_mean"].replace(0, np.nan)).clip(0, 14).fillna(
        (onhand / df["forecast_mean"].replace(0, np.nan)).clip(0, 14).median()
    )

    # Lagged forecast and percent change
    df = df.sort_values(["asin", "week_monday"])
    df["forecast_lag1"]   = df.groupby("asin")["forecast_mean"].shift(1)
    df["forecast_change"] = (
        (df["forecast_mean"] - df["forecast_lag1"]) /
        (df["forecast_lag1"].abs() + 1)
    ).fillna(0)

    df = build_holiday_flags(df)

    cols = [
        "asin", "week_monday", "po_quantity", "forecast_mean",
        "forecast_lag1", "forecast_change", "woc",
        "evt_halloween", "evt_holiday", "evt_cyber", "evt_christmas",
    ]
    return df[[c for c in cols if c in df.columns]].reset_index(drop=True)


# ------------------------------------------------------------------
# MODEL A — PO QUANTITY (LINEAR REGRESSION)
# ------------------------------------------------------------------

FEAT_COLS_QTY = [
    "forecast_mean", "woc",
    "evt_halloween", "evt_holiday", "evt_cyber", "evt_christmas",
]


def train_quantity_model(panel: pd.DataFrame) -> LinearRegression:
    """Train a linear regression model to predict PO quantity."""
    train = panel[panel["po_quantity"].notna()].fillna(0)
    X = train[FEAT_COLS_QTY].values
    y = train["po_quantity"].values
    model = LinearRegression()
    model.fit(X, y)
    return model


def evaluate_quantity_model(model: LinearRegression, panel: pd.DataFrame) -> None:
    """Report MAE and MAPE of the quantity model on training data."""
    train = panel[panel["po_quantity"].notna()].fillna(0)
    X     = train[FEAT_COLS_QTY].fillna(0)
    y     = train["po_quantity"]
    y_hat = model.predict(X)

    mae  = mean_absolute_error(y, y_hat)
    nonzero = y != 0
    mape = mean_absolute_percentage_error(y[nonzero], y_hat[nonzero])

    print(f"\n  Quantity Model (Linear Regression)")
    print(f"    MAE  : {mae:.2f} units")
    print(f"    MAPE : {mape:.2%} (on non-zero PO weeks)")


# ------------------------------------------------------------------
# MODEL B — PO OCCURRENCE (LOGISTIC REGRESSION + RANDOM FOREST)
# ------------------------------------------------------------------

FEAT_COLS_OCC = [
    "forecast_mean", "forecast_change", "woc",
    "evt_halloween", "evt_holiday", "evt_cyber", "evt_christmas",
]


def build_occurrence_dataset(panel: pd.DataFrame) -> tuple:
    """
    Build classification dataset where the target is whether
    Amazon will place a PO in the NEXT week.
    """
    df = panel.copy()
    df["po_flag"] = (df["po_quantity"] > 0).astype(int)
    df["po_flag_next_week"] = (
        df.sort_values(["asin", "week_monday"])
          .groupby("asin")["po_flag"]
          .shift(-1)
    )
    df = df.dropna(subset=["po_flag_next_week"]).copy()
    df["po_flag_next_week"] = df["po_flag_next_week"].astype(int)

    df[FEAT_COLS_OCC] = df[FEAT_COLS_OCC].fillna(0)
    X = df[FEAT_COLS_OCC]
    y = df["po_flag_next_week"]
    return train_test_split(X, y, test_size=0.25, shuffle=False)


def train_occurrence_models(X_train, y_train) -> tuple:
    """Train logistic regression and random forest classifiers."""
    logit = LogisticRegression(max_iter=1000)
    logit.fit(X_train, y_train)

    rf = RandomForestClassifier(
        n_estimators=250, max_depth=8,
        min_samples_split=50, class_weight="balanced", random_state=42,
    )
    rf.fit(X_train, y_train)
    return logit, rf


def evaluate_occurrence_model(model, name: str, X_test, y_test,
                               threshold: float = OCC_THRESHOLD) -> None:
    """Report Precision, Recall, F1, and AUC for a classifier."""
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)
    print(f"\n  {name} (threshold={threshold})")
    print(f"    Precision : {precision_score(y_test, pred):.3f}")
    print(f"    Recall    : {recall_score(y_test, pred):.3f}")
    print(f"    F1 Score  : {f1_score(y_test, pred):.3f}")
    print(f"    AUC       : {roc_auc_score(y_test, prob):.3f}")


# ------------------------------------------------------------------
# COMBINED FORECAST — OCCURRENCE × QUANTITY
# ------------------------------------------------------------------

def build_event_flags_for_date(target_date: str) -> dict:
    """Compute holiday event flags for a given target date."""
    d    = pd.to_datetime(target_date)
    year = d.year

    halloween    = pd.to_datetime(f"{year}-10-31")
    nov_days     = pd.date_range(f"{year}-11-01", f"{year}-11-30", freq="D")
    thanksgiving = nov_days[nov_days.weekday == 3][3]
    cyber_monday = thanksgiving + pd.Timedelta(days=4)
    christmas    = pd.to_datetime(f"{year}-12-25")

    def to_monday(x):
        return x - pd.to_timedelta(x.weekday(), unit="D")

    week_mon = to_monday(d)
    return {
        "evt_halloween": int(week_mon == to_monday(halloween)),
        "evt_holiday":   int(week_mon == to_monday(thanksgiving)),
        "evt_cyber":     int(week_mon == to_monday(cyber_monday)),
        "evt_christmas": int(week_mon == to_monday(christmas)),
    }


def predict_po_for_week(
    panel: pd.DataFrame,
    model_occ, model_qty,
    target_date: str,
    threshold: float = OCC_THRESHOLD,
) -> pd.DataFrame:
    """
    Generate a combined PO forecast for all ASINs for a target week.

    Returns a DataFrame with:
        - asin, target_week_monday
        - forecast_mean, woc
        - po_probability, expected_po_qty
        - reorder_flag (WoC ≤ 6)
        - holiday event flags
    """
    target = pd.to_datetime(target_date)
    week_mon = target - pd.to_timedelta(target.weekday(), unit="D")
    events = build_event_flags_for_date(target_date)

    last = (
        panel.sort_values(["asin", "week_monday"])
             .groupby("asin")
             .tail(1)
             .reset_index(drop=True)
    )

    rows = []
    for _, r in last.iterrows():
        lag = r["forecast_mean"]
        fc_change = (lag - lag) / (abs(lag) + 1)   # no change assumed for future week
        rows.append({
            "asin":              r["asin"],
            "week_monday":       week_mon,
            "forecast_mean":     r["forecast_mean"],
            "woc":               r["woc"],
            "forecast_change":   fc_change,
            **events,
        })

    future = pd.DataFrame(rows)

    X_occ = future[FEAT_COLS_OCC].fillna(0)
    X_qty = future[FEAT_COLS_QTY].fillna(0)

    po_prob    = model_occ.predict_proba(X_occ)[:, 1]
    po_flag    = (po_prob >= threshold).astype(int)
    qty_pred   = np.maximum(model_qty.predict(X_qty.values), 0)
    exp_po_qty = qty_pred * po_flag

    out = future.copy()
    out["po_probability"]  = po_prob
    out["expected_po_qty"] = exp_po_qty
    out["reorder_flag"]    = (out["woc"] <= 6).astype(int)
    out["target_week"]     = week_mon

    return (
        out[[
            "asin", "target_week", "forecast_mean", "woc",
            "po_probability", "expected_po_qty", "reorder_flag",
            "evt_halloween", "evt_holiday", "evt_cyber", "evt_christmas",
        ]]
        .sort_values("po_probability", ascending=False)
        .reset_index(drop=True)
    )


def build_projection_table(
    panel: pd.DataFrame,
    model_occ, model_qty,
    week_list: list,
    threshold: float = OCC_THRESHOLD,
) -> pd.DataFrame:
    """
    Generate a multi-week PO projection table for all ASINs.
    Returns a clean asin / week_start_date / expected_po_qty table.
    """
    frames = []
    for wk in week_list:
        forecast = predict_po_for_week(panel, model_occ, model_qty, wk, threshold)
        simple = (
            forecast[["asin", "target_week", "expected_po_qty"]]
            .rename(columns={"target_week": "week_start_date"})
        )
        frames.append(simple)

    proj = pd.concat(frames, ignore_index=True)
    proj["expected_po_qty"] = proj["expected_po_qty"].round()
    return proj.sort_values(["asin", "week_start_date"]).reset_index(drop=True)


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Franklin Sports | Weekly PO Forecasting (Objective 3)")
    print("=" * 60)

    print("\n[1/4] Loading and building modeling panel...")
    df_raw = pd.read_excel(MERGED_PATH)

    # Override with validation PO data if available
    if VAL_PATH.exists():
        val = pd.read_excel(VAL_PATH)
        val["date"] = pd.to_datetime(val.get("date", val.get("po_request_date")), errors="coerce")
        val["week_monday"] = val["date"] - pd.to_timedelta(val["date"].dt.weekday, unit="D")
        val_weekly = (
            val.groupby(["asin", "week_monday"], as_index=False)["quantity"]
               .sum()
               .rename(columns={"quantity": "po_quantity_val"})
        )
        df_raw["week_monday"] = pd.to_datetime(
            df_raw.get("start_date", df_raw.get("week_monday")), errors="coerce"
        )
        df_raw["week_monday"] -= pd.to_timedelta(df_raw["week_monday"].dt.weekday, unit="D")
        df_raw = df_raw.merge(val_weekly, on=["asin", "week_monday"], how="left")
        df_raw["po_quantity"] = (
            df_raw["po_quantity_val"]
            .fillna(df_raw.get("po_quantity", pd.Series(0, index=df_raw.index)))
            .fillna(0)
        )
        print("  Validation PO data merged.")

    panel = build_panel(df_raw)
    print(f"  Panel shape: {panel.shape}")

    print("\n[2/4] Training quantity model (Linear Regression)...")
    model_qty = train_quantity_model(panel)
    evaluate_quantity_model(model_qty, panel)

    print("\n[3/4] Training occurrence models...")
    X_tr, X_te, y_tr, y_te = build_occurrence_dataset(panel)
    model_logit, model_rf = train_occurrence_models(X_tr, y_tr)

    print("\n  Occurrence Model Performance:")
    evaluate_occurrence_model(model_logit, "Logistic Regression", X_te, y_te)
    evaluate_occurrence_model(model_rf,    "Random Forest",       X_te, y_te)

    print("\n[4/4] Generating example forecasts...")
    print("\n  Non-holiday week (2025-10-06):")
    forecast_generic = predict_po_for_week(panel, model_rf, model_qty, "2025-10-06")
    print(forecast_generic.head(10).to_string(index=False))

    print("\n  Halloween week (2025-10-31):")
    forecast_halloween = predict_po_for_week(panel, model_rf, model_qty, "2025-10-31")
    print(forecast_halloween.head(10).to_string(index=False))

    print("\n  Multi-week projection table (Sep–Oct 2025):")
    weeks = ["2025-09-15", "2025-09-22", "2025-09-29", "2025-10-06"]
    proj_table = build_projection_table(panel, model_rf, model_qty, weeks)
    print(proj_table.head(12).to_string(index=False))

    print("\nObjective 3 complete.")
    return panel, model_rf, model_qty


if __name__ == "__main__":
    main()
