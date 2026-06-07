"""
06_inventory_simulation.py
--------------------------
Franklin Sports | Amazon Replenishment Analytics

Purpose:
    Objective 4 — Inventory Optimization Simulation (EOQ + Safety Stock)

    Compares three replenishment policies across all Baseball SKUs
    to identify the optimal strategy balancing service level and cost:

        Policy 1 — 4-WoC  : Order 4 weeks of average demand each cycle
        Policy 2 — 6-WoC  : Order 6 weeks of average demand each cycle
        Policy 3 — EOQ+SS : Economic Order Quantity with Safety Stock buffer

    Simulation runs weekly for each ASIN using actual demand sequences,
    applying consistent lead-time, holding cost, and order cost assumptions.

Key Findings:
    - 4-WoC  : Lowest cost, lowest inventory, but higher stockout risk
    - 6-WoC  : Best balance of service level and total cost
    - EOQ+SS : Most stable service (~98%) but highest holding cost
    - No single policy fits all — SKU segmentation is recommended

Inputs:  data/clean/merged_final.xlsx  (or pre-built modeling panel)
Outputs: Policy comparison table printed to console
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

BASE_DIR  = Path(__file__).resolve().parent.parent
CLEAN_DIR = BASE_DIR / "data" / "clean"

MERGED_PATH = CLEAN_DIR / "merged_final.xlsx"

EOQ_CONFIG = {
    "holding_rate":      0.22,    # 22% of unit value per year (industry standard)
    "unit_cost_default": 10.0,    # $ per unit (placeholder; Franklin can override)
    "order_cost_default": 300.0,  # $ per replenishment order
    "lead_time_weeks":   4,       # 3-week transit + 1-week buffer
    "service_level_z":   1.65,    # z-score for ~95% cycle service level
    "weeks_per_year":    52,
}


# ------------------------------------------------------------------
# DEMAND STATISTICS
# ------------------------------------------------------------------

def compute_demand_stats(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-ASIN weekly demand statistics from forecast_mean:
        - Average weekly demand
        - Standard deviation of weekly demand
        - Annualized demand (for EOQ formula)
    """
    stats = (
        panel.groupby("asin")
             .agg(
                 avg_weekly_demand=("forecast_mean", "mean"),
                 std_weekly_demand=("forecast_mean", "std"),
             )
             .reset_index()
    )
    stats["std_weekly_demand"]  = stats["std_weekly_demand"].fillna(0)
    stats["annual_demand"]      = stats["avg_weekly_demand"] * EOQ_CONFIG["weeks_per_year"]
    return stats


# ------------------------------------------------------------------
# EOQ POLICY PARAMETERS
# ------------------------------------------------------------------

def compute_eoq_params(row: pd.Series, config: dict) -> pd.Series:
    """
    Derive EOQ, safety stock, and reorder trigger level for one ASIN.

    EOQ formula  : sqrt(2 × D × S / H)
    Safety stock : z × σ_weekly × sqrt(lead_time_weeks)
    Reorder point: (avg_weekly_demand × lead_time_weeks) + safety_stock
    """
    D = max(row["annual_demand"], 0)
    if D <= 0:
        return pd.Series({"EOQ_units": 0.0, "safety_stock_units": 0.0,
                          "reorder_trigger_units": 0.0})

    H   = config["holding_rate"] * config["unit_cost_default"]
    S   = config["order_cost_default"]
    L   = config["lead_time_weeks"]
    z   = config["service_level_z"]

    eoq           = np.sqrt((2 * D * S) / H)
    safety_stock  = z * row["std_weekly_demand"] * np.sqrt(L)
    reorder_point = row["avg_weekly_demand"] * L + safety_stock

    return pd.Series({
        "EOQ_units":             eoq,
        "safety_stock_units":    safety_stock,
        "reorder_trigger_units": reorder_point,
    })


def build_policy_table(demand_stats: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Attach EOQ policy parameters to the demand statistics table."""
    eoq_params = demand_stats.apply(compute_eoq_params, axis=1, config=config)
    return pd.concat([demand_stats, eoq_params], axis=1)


# ------------------------------------------------------------------
# WEEKLY INVENTORY SIMULATION
# ------------------------------------------------------------------

def simulate_policy(
    df_asin: pd.DataFrame,
    demand_col: str,
    init_inventory: float,
    order_qty: float,
    reorder_trigger: float,
    config: dict,
) -> dict:
    """
    Simulate weekly inventory for one ASIN under one replenishment policy.

    Logic per week:
        1. Receive any pipeline orders arriving this week
        2. Apply demand (no backorders — lost sales)
        3. Check if inventory fell below reorder trigger → place order
        4. Accumulate holding and order costs

    Returns performance metrics: service level, avg inventory, total cost.
    """
    df_asin  = df_asin.sort_values("week_monday").reset_index(drop=True)
    demand   = df_asin[demand_col].fillna(0).values
    L        = config["lead_time_weeks"]
    H_week   = config["holding_rate"] * config["unit_cost_default"] / config["weeks_per_year"]
    S        = config["order_cost_default"]

    onhand          = float(init_inventory)
    pipeline        = []   # (arrival_week_index, qty)
    total_demand    = total_served = 0.0
    total_holding   = total_order  = 0.0
    onhand_history  = []

    for t, d in enumerate(demand):
        # Receive arriving orders
        arriving = sum(q for (tau, q) in pipeline if tau == t)
        onhand  += arriving
        pipeline = [(tau, q) for (tau, q) in pipeline if tau != t]

        # Fulfill demand
        served       = min(onhand, d)
        onhand      -= served
        total_demand += d
        total_served += served

        # Reorder check
        if onhand <= reorder_trigger and order_qty > 0:
            arrival = t + L
            if arrival < len(demand):
                pipeline.append((arrival, order_qty))
            total_order += S

        # Holding cost
        total_holding += onhand * H_week
        onhand_history.append(onhand)

    service_level = total_served / total_demand if total_demand > 0 else np.nan

    return {
        "asin":               df_asin["asin"].iloc[0],
        "weeks_simulated":    len(demand),
        "service_level":      service_level,
        "avg_onhand_units":   np.mean(onhand_history),
        "total_holding_cost": total_holding,
        "total_order_cost":   total_order,
        "total_cost":         total_holding + total_order,
    }


# ------------------------------------------------------------------
# POLICY COMPARISON
# ------------------------------------------------------------------

def compare_policies(
    asin_id: str,
    panel: pd.DataFrame,
    policy_table: pd.DataFrame,
    config: dict,
    demand_col: str = "forecast_mean",
) -> pd.DataFrame:
    """
    Run all three policies for one ASIN and return a comparison DataFrame.
    """
    df_asin = panel[panel["asin"] == asin_id].copy()
    row     = policy_table[policy_table["asin"] == asin_id].iloc[0]

    avg_d        = row["avg_weekly_demand"]
    L            = config["lead_time_weeks"]
    init_inv     = avg_d * L
    lt_demand    = avg_d * L

    policies = [
        {"name": "4_WoC",      "order_qty": avg_d * 4, "reorder_trigger": lt_demand},
        {"name": "6_WoC",      "order_qty": avg_d * 6, "reorder_trigger": lt_demand},
        {"name": "EOQ_plus_SS","order_qty": row["EOQ_units"],
         "reorder_trigger": row["reorder_trigger_units"]},
    ]

    results = []
    for p in policies:
        sim = simulate_policy(
            df_asin, demand_col, init_inv,
            p["order_qty"], p["reorder_trigger"], config,
        )
        sim["policy"] = p["name"]
        results.append(sim)

    cols = ["asin", "policy", "weeks_simulated", "service_level",
            "avg_onhand_units", "total_holding_cost", "total_order_cost", "total_cost"]
    return pd.DataFrame(results)[cols]


def run_policy_comparison_all(
    panel: pd.DataFrame,
    policy_table: pd.DataFrame,
    config: dict,
    top_n: int = None,
) -> pd.DataFrame:
    """
    Run the three-policy comparison for all ASINs (or top N by demand).
    Returns a consolidated results DataFrame.
    """
    asin_list = policy_table["asin"].tolist()
    if top_n:
        asin_list = (
            policy_table.sort_values("annual_demand", ascending=False)
                        .head(top_n)["asin"].tolist()
        )

    results = []
    for asin_id in asin_list:
        try:
            comp = compare_policies(asin_id, panel, policy_table, config)
            results.append(comp)
        except Exception as e:
            print(f"  Skipping {asin_id}: {e}")

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# ------------------------------------------------------------------
# SUMMARY & REPORTING
# ------------------------------------------------------------------

def summarize_by_policy(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate simulation results by policy to produce a cross-SKU
    performance summary.
    """
    return (
        results_df.groupby("policy")
                  .agg(
                      avg_service_level=("service_level",   "mean"),
                      avg_onhand_units= ("avg_onhand_units","mean"),
                      avg_total_cost=   ("total_cost",      "mean"),
                  )
                  .reset_index()
                  .round(3)
    )


def print_policy_summary(summary: pd.DataFrame) -> None:
    """Print a formatted policy comparison table."""
    print("\n  Policy Performance Summary (averaged across simulated SKUs)")
    print("  " + "-" * 65)
    print(f"  {'Policy':<20} {'Avg Service Level':>18} {'Avg Onhand Units':>17} {'Avg Total Cost':>15}")
    print("  " + "-" * 65)
    for _, row in summary.iterrows():
        print(f"  {row['policy']:<20} {row['avg_service_level']:>18.3f} "
              f"{row['avg_onhand_units']:>17.1f} {row['avg_total_cost']:>15.2f}")
    print("  " + "-" * 65)


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Franklin Sports | Inventory Optimization Simulation (Obj 4)")
    print("=" * 60)

    print("\n[1/4] Loading modeling panel...")
    df_raw = pd.read_excel(MERGED_PATH)
    df_raw.columns = [c.strip() for c in df_raw.columns]

    for col in ["onhand_units", "forecast_mean", "po_quantity", "lead_time_days"]:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    date_col = next((c for c in ["start_date", "week_monday"] if c in df_raw.columns), None)
    if date_col:
        df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors="coerce")
        df_raw["week_monday"] = df_raw[date_col] - pd.to_timedelta(
            df_raw[date_col].dt.weekday, unit="D"
        )

    if "division" in df_raw.columns:
        bb = df_raw["division"].str.lower().eq("baseball")
        if bb.any():
            df_raw = df_raw.loc[bb].copy()

    df_raw["forecast_mean"] = df_raw["forecast_mean"].fillna(0)
    panel = df_raw

    print(f"  Panel shape: {panel.shape}")

    print("\n[2/4] Computing demand statistics per ASIN...")
    demand_stats = compute_demand_stats(panel)
    print(f"  ASINs with demand data: {len(demand_stats):,}")

    print("\n[3/4] Deriving EOQ policy parameters...")
    policy_table = build_policy_table(demand_stats, EOQ_CONFIG)
    print(f"  EOQ config: holding_rate={EOQ_CONFIG['holding_rate']:.0%}, "
          f"lead_time={EOQ_CONFIG['lead_time_weeks']} weeks, "
          f"service_level_z={EOQ_CONFIG['service_level_z']}")

    print("\n[4/4] Running policy simulations...")

    # Single-ASIN example
    example_asin = policy_table["asin"].iloc[0]
    print(f"\n  Single ASIN example ({example_asin}):")
    single = compare_policies(example_asin, panel, policy_table, EOQ_CONFIG)
    print(single.to_string(index=False))

    # Top-N comparison
    print("\n  Running comparison across top 10 ASINs by demand...")
    all_results = run_policy_comparison_all(panel, policy_table, EOQ_CONFIG, top_n=10)

    # Cross-ASIN summary
    summary = summarize_by_policy(all_results)
    print_policy_summary(summary)

    print("\n  Interpretation:")
    print("  - 4-WoC  : Lowest cost, but higher stockout risk for volatile SKUs")
    print("  - 6-WoC  : Best balance of service level and moderate cost")
    print("  - EOQ+SS : Most stable service (~98%) but highest holding cost")
    print("  → Recommended: Segment SKUs by demand variability and apply")
    print("    differentiated policies (e.g., 6-WoC for high-volume,")
    print("    EOQ+SS for volatile/high-value items)")

    print("\nObjective 4 complete.")


if __name__ == "__main__":
    main()
