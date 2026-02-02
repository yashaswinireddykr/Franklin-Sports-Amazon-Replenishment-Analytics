from dataclasses import dataclass
import numpy as np
import pandas as pd


# =========================
# Simulation parameters
# =========================
@dataclass
class SimParams:
    forecast_horizon_weeks: int
    lead_time_weeks: int
    review_period_weeks: int
    service_level: float
    safety_stock_factor: float
    n_simulations: int


# =========================
# Inventory simulation
# =========================
def simulate_inventory_po(
    df_asin: pd.DataFrame,
    params: SimParams,
):
    """
    NDA-safe inventory & PO simulation.
    Uses masked demand signals only.
    """

    # --- Basic demand proxy (masked) ---
    # Use onhand_units as a scale proxy (safe, non-proprietary)
    avg_weekly_demand = max(df_asin["onhand_units"].mean() / 12, 1)

    # --- Monte Carlo demand simulation ---
    demand_sims = np.random.normal(
        loc=avg_weekly_demand,
        scale=avg_weekly_demand * 0.3,
        size=(params.n_simulations, params.forecast_horizon_weeks),
    )
    demand_sims = np.clip(demand_sims, 0, None)

    total_demand = demand_sims.sum(axis=1)

    # --- Safety stock ---
    z = params.safety_stock_factor
    safety_stock = z * np.std(total_demand)

    # --- Reorder point ---
    reorder_point = (
        avg_weekly_demand * params.lead_time_weeks
        + safety_stock
    )

    # --- Current inventory proxy ---
    current_inventory = df_asin["onhand_units"].iloc[0]

    # --- Recommended PO ---
    recommended_po = max(
        reorder_point - current_inventory, 0
    )

    # --- Weeks of cover ---
    weeks_of_cover = current_inventory / avg_weekly_demand

    # --- Stockout risk ---
    stockout_risk = float(
        np.mean(total_demand > current_inventory)
    )

    return {
        "recommended_po_qty": int(round(recommended_po)),
        "weeks_of_cover": round(weeks_of_cover, 1),
        "stockout_risk": round(stockout_risk * 100, 1),
    }

