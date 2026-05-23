from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.compare_capacity_plans import apply_scenario, hard_capacity_allocate, make_plan
from src.config import load_config
from src.data.loaders import load_raw_data
from src.models.problem1_markov_demand import STATE_CN, MarkovDemandModel
from src.models.problem2_cmclp_bilevel import LocationCapacityOptimizer
from src.utils.metrics import price_satisfaction


SUBSIDY_CAP_DAILY = {"小型": 1000.0, "中型": 1800.0, "大型": 2600.0}


@dataclass
class CapacityHardPricingResult:
    station_prices: pd.DataFrame
    station_finance: pd.DataFrame
    community_satisfaction: pd.DataFrame
    demand_release: pd.DataFrame
    scenario_summary: dict[str, Any]


def recommended_specs(scenario: str) -> dict[str, str]:
    if scenario == "budget_140":
        return {"A": "小型", "B": "大型", "G": "大型", "I": "中型"}
    return {"C": "小型", "E": "小型", "G": "小型", "I": "小型", "J": "大型"}


def build_state_records(raw, p1_outputs) -> list[dict[str, Any]]:
    freq = raw.demand_freq.set_index("service")
    pop5 = p1_outputs.year5_population.set_index("community")
    services = list(raw.service_cost["service"])
    records = []
    for community, c in pop5.iterrows():
        for state_col, state_name in STATE_CN.items():
            records.append(
                {
                    "community": community,
                    "elder_type": state_name,
                    "n": float(c[state_col]),
                    "income": float(c["monthly_income"]),
                    "cap_ratio": float(raw.consumption_caps[state_name]),
                    "freq": {s: float(freq.loc[s, state_name]) for s in services},
                }
            )
    return records


def allocation_shares(assignment: pd.DataFrame) -> dict[tuple[str, str], float]:
    shares = {}
    for r in assignment.itertuples():
        if float(r.daily_demand) <= 0:
            continue
        for station in str(r.actual_stations).split(","):
            station = station.strip()
            if not station:
                continue
        # The aggregate assignment file does not preserve station shares. Use the
        # dominant station for fixed-price optimization; capacity is enforced again
        # at the station level below.
        if isinstance(r.dominant_station, str) and r.dominant_station:
            shares[(r.community, r.dominant_station)] = float(r.served_daily_demand) / float(r.daily_demand)
    return shares


def service_price_candidates(base_price: float, emergency: bool) -> list[float]:
    if emergency:
        return [0.0]
    # A compact grid keeps the capacity-hard P3/P4 pipeline runnable while still
    # testing every satisfaction tier boundary in Attachment 5.
    multipliers = [0.60, 0.80, 1.00, 1.10, 1.20]
    return sorted({round(base_price * m, 4) for m in multipliers})


def demand_under_prices(
    records: list[dict[str, Any]],
    services: list[str],
    prices: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    release_base = []
    for rec in records:
        per_cost = sum(rec["freq"][s] * float(prices[s]) for s in services)
        cap = rec["income"] * rec["cap_ratio"]
        scale = min(1.0, cap / per_cost) if per_cost > 0 else 1.0
        for s in services:
            rows.append(
                {
                    "community": rec["community"],
                    "elder_type": rec["elder_type"],
                    "service": s,
                    "monthly_demand": rec["n"] * rec["freq"][s] * scale,
                }
            )
        release_base.append(
            {
                "community": rec["community"],
                "elder_type": rec["elder_type"],
                "price_sensitive_scale": scale,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(release_base)


def station_finance(
    station: str,
    scale: str,
    plan_info: dict[str, Any],
    station_communities: list[str],
    community_share: dict[tuple[str, str], float],
    records: list[dict[str, Any]],
    services: list[str],
    emergency_service: str,
    service_cost: pd.DataFrame,
    base_budget: pd.DataFrame,
    theoretical: pd.DataFrame,
    prices: dict[str, float],
    days_per_month: float,
    days_per_year: float,
) -> dict[str, Any]:
    priced_demand, _ = demand_under_prices(records, services, prices)
    priced_demand = priced_demand[priced_demand["community"].isin(station_communities)].copy()
    priced_demand["station_share"] = priced_demand["community"].map(lambda c: community_share.get((c, station), 0.0))
    priced_demand["allocated_monthly_demand"] = priced_demand["monthly_demand"] * priced_demand["station_share"]

    raw_monthly_total = float(priced_demand["allocated_monthly_demand"].sum())
    raw_daily_total = raw_monthly_total / days_per_month
    capacity = float(plan_info["capacity"])
    capacity_scale = min(1.0, capacity / raw_daily_total) if raw_daily_total > 1e-9 else 1.0
    served = priced_demand.copy()
    served["served_monthly_demand"] = served["allocated_monthly_demand"] * capacity_scale

    util = (float(served["served_monthly_demand"].sum()) / days_per_month) / capacity if capacity > 0 else 0.0
    # The capacity-hard allocation has already selected distance-feasible service.
    # Use the station's realized utilization for response and the station-level
    # demand-weighted price score for effective service.
    response_score = 1.0 if util <= 0.60 else 0.93 if util <= 0.75 else 0.85 if util <= 0.85 else 0.72 if util <= 0.95 else 0.60
    price_scores = {s: price_satisfaction(float(prices[s]), float(service_cost.loc[s, "base_price"])) for s in services}
    served["price_score"] = served["service"].map(price_scores)
    weighted_price_satisfaction = (
        float((served["served_monthly_demand"] * served["price_score"]).sum() / served["served_monthly_demand"].sum())
        if float(served["served_monthly_demand"].sum()) > 1e-9
        else 1.0
    )
    effective_multiplier = 0.3 * response_score + 0.5 * weighted_price_satisfaction + 0.2 * 0.9
    served["effective_monthly_demand"] = served["served_monthly_demand"] * effective_multiplier

    cost = service_cost
    served["price"] = served["service"].map(prices)
    served["direct_cost"] = served["service"].map(cost["direct_cost"])
    annual_revenue = float((served["served_monthly_demand"] * served["price"] * 12).sum())
    annual_direct = float((served["served_monthly_demand"] * served["direct_cost"] * 12).sum())

    non_emergency_eff_daily = (
        float(served.loc[served["service"] != emergency_service, "effective_monthly_demand"].sum()) / days_per_month
    )
    annual_subsidy = min(2.0 * non_emergency_eff_daily, SUBSIDY_CAP_DAILY[scale]) * days_per_year
    fixed = float(plan_info["fixed_daily_cost"]) * days_per_year
    depreciation = float(plan_info["build_cost_wan"]) * 10000 / 20
    annual_operating_cost = fixed + depreciation
    service_profit = annual_revenue - annual_direct
    profit_after_subsidy = service_profit + annual_subsidy - annual_operating_cost
    profit_rate = profit_after_subsidy / annual_operating_cost if annual_operating_cost > 0 else 0.0

    base_station = base_budget[base_budget["community"].isin(station_communities)].copy()
    base_station["station_share"] = base_station["community"].map(lambda c: community_share.get((c, station), 0.0))
    base_station_total = float((base_station["monthly_demand"] * base_station["station_share"]).sum())
    theory_station = theoretical[theoretical["community"].isin(station_communities)].copy()
    theory_station["station_share"] = theory_station["community"].map(lambda c: community_share.get((c, station), 0.0))
    theory_station_total = float((theory_station["monthly_demand"] * theory_station["station_share"]).sum())
    released = raw_monthly_total - base_station_total
    release_index = released / (theory_station_total - base_station_total) if theory_station_total > base_station_total + 1e-9 else 0.0

    return {
        "annual_revenue": annual_revenue,
        "annual_direct_cost": annual_direct,
        "service_profit": service_profit,
        "annual_subsidy": float(annual_subsidy),
        "annual_operating_cost": float(annual_operating_cost),
        "annual_profit_after_subsidy": float(profit_after_subsidy),
        "profit_rate": float(profit_rate),
        "pricing_feasible": int(0.0 <= profit_rate <= 0.08),
        "utilization_after_pricing": float(util),
        "response_score_after_pricing": float(response_score),
        "weighted_price_satisfaction": float(weighted_price_satisfaction),
        "effective_multiplier": float(effective_multiplier),
        "raw_monthly_demand_after_pricing": float(raw_monthly_total),
        "served_monthly_demand_after_capacity": float(served["served_monthly_demand"].sum()),
        "unserved_released_monthly_demand": float(max(0.0, raw_monthly_total - served["served_monthly_demand"].sum())),
        "latent_demand_release_index": float(max(0.0, min(1.0, release_index))),
    }


def optimize_station_prices(
    station: str,
    scale: str,
    plan_info: dict[str, Any],
    station_communities: list[str],
    community_share: dict[tuple[str, str], float],
    records: list[dict[str, Any]],
    services: list[str],
    emergency_service: str,
    service_cost: pd.DataFrame,
    base_budget: pd.DataFrame,
    theoretical: pd.DataFrame,
    days_per_month: float,
    days_per_year: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    non_emergency = [s for s in services if s != emergency_service]
    candidates = [service_price_candidates(float(service_cost.loc[s, "base_price"]), False) for s in non_emergency]
    best_obj = None
    best_prices = None
    best_finance = None
    evaluated = 0
    for combo in product(*candidates):
        evaluated += 1
        prices = {s: float(p) for s, p in zip(non_emergency, combo)}
        prices[emergency_service] = 0.0
        finance = station_finance(
            station,
            scale,
            plan_info,
            station_communities,
            community_share,
            records,
            services,
            emergency_service,
            service_cost,
            base_budget,
            theoretical,
            prices,
            days_per_month,
            days_per_year,
        )
        feasible = bool(finance["pricing_feasible"])
        if finance["profit_rate"] < 0:
            violation = -finance["profit_rate"]
        elif finance["profit_rate"] > 0.08:
            violation = finance["profit_rate"] - 0.08
        else:
            violation = 0.0
        avg_price = float(np.mean([prices[s] for s in non_emergency]))
        obj = (
            1 if feasible else 0,
            -violation,
            finance["weighted_price_satisfaction"],
            -abs(finance["profit_rate"] - 0.08),
            finance["latent_demand_release_index"],
            -avg_price,
        )
        if best_obj is None or obj > best_obj:
            best_obj = obj
            best_prices = prices
            best_finance = finance
    assert best_prices is not None and best_finance is not None
    best_finance = dict(best_finance)
    best_finance["price_combinations_evaluated"] = evaluated
    return best_prices, best_finance


def run_scenario(scenario: str, output_dir: Path) -> CapacityHardPricingResult:
    cfg0 = load_config("configs/default.yaml")
    raw0 = load_raw_data(cfg0)
    raw, cfg = apply_scenario(raw0, cfg0, scenario)
    p1 = MarkovDemandModel(raw, cfg).run()
    helper = LocationCapacityOptimizer(raw, cfg, p1)
    spec = recommended_specs(scenario)
    plan = make_plan(helper.scale_map, spec)
    p2 = hard_capacity_allocate(raw, cfg, p1, plan, f"{scenario}_recommended", min_community_fulfillment=0.8)

    services = list(raw.service_cost["service"])
    emergency = cfg["problem3"]["emergency_service_name"]
    service_cost = raw.service_cost.set_index("service")
    records = build_state_records(raw, p1)
    shares = allocation_shares(p2.assignment)
    days_per_month = float(cfg["problem2"]["days_per_month"])
    days_per_year = float(cfg["problem2"]["days_per_year"])

    price_rows = []
    finance_rows = []
    release_rows = []
    for station, info in plan.items():
        station_communities = p2.assignment.loc[p2.assignment["dominant_station"] == station, "community"].dropna().tolist()
        prices, finance = optimize_station_prices(
            station,
            str(info["scale"]),
            info,
            station_communities,
            shares,
            records,
            services,
            emergency,
            service_cost,
            p1.budgeted_monthly_demand,
            p1.theoretical_monthly_demand,
            days_per_month,
            days_per_year,
        )
        for service in services:
            price_rows.append(
                {
                    "scenario": scenario,
                    "station": station,
                    "service": service,
                    "base_price": float(service_cost.loc[service, "base_price"]),
                    "optimized_price": float(prices[service]),
                    "price_satisfaction": price_satisfaction(float(prices[service]), float(service_cost.loc[service, "base_price"])),
                }
            )
        finance_rows.append({"scenario": scenario, "station": station, "scale": info["scale"], **finance})
        release_rows.append(
            {
                "scenario": scenario,
                "station": station,
                "latent_demand_release_index": finance["latent_demand_release_index"],
                "unserved_released_monthly_demand": finance["unserved_released_monthly_demand"],
            }
        )

    station_prices = pd.DataFrame(price_rows)
    station_finance = pd.DataFrame(finance_rows)
    demand_release = pd.DataFrame(release_rows)
    community_satisfaction = p2.assignment.copy()
    community_satisfaction.insert(0, "scenario", scenario)

    summary = {
        "scenario": scenario,
        "locations": ",".join(plan.keys()),
        "scales": ",".join(str(plan[j]["scale"]) for j in plan.keys()),
        **p2.metrics,
        "annual_subsidy_yuan_problem3": float(station_finance["annual_subsidy"].sum()),
        "annual_profit_after_subsidy_problem3": float(station_finance["annual_profit_after_subsidy"].sum()),
        "avg_profit_rate_problem3": float(station_finance["profit_rate"].mean()),
        "min_profit_rate_problem3": float(station_finance["profit_rate"].min()),
        "pricing_feasible": int(station_finance["pricing_feasible"].min()),
        "avg_price_satisfaction_problem3": float(station_prices["price_satisfaction"].mean()),
    }

    prefix = output_dir / f"{scenario}_capacity_hard"
    p2.assignment.to_csv(prefix.with_name(prefix.name + "_problem2_assignment.csv"), index=False, encoding="utf-8-sig")
    p2.station_status.to_csv(prefix.with_name(prefix.name + "_problem2_station_status.csv"), index=False, encoding="utf-8-sig")
    p2.station_profit.to_csv(prefix.with_name(prefix.name + "_problem2_profit_without_subsidy.csv"), index=False, encoding="utf-8-sig")
    station_prices.to_csv(prefix.with_name(prefix.name + "_problem3_station_prices.csv"), index=False, encoding="utf-8-sig")
    station_finance.to_csv(prefix.with_name(prefix.name + "_problem3_station_finance.csv"), index=False, encoding="utf-8-sig")
    community_satisfaction.to_csv(prefix.with_name(prefix.name + "_problem3_community_satisfaction.csv"), index=False, encoding="utf-8-sig")
    demand_release.to_csv(prefix.with_name(prefix.name + "_problem3_demand_release.csv"), index=False, encoding="utf-8-sig")
    return CapacityHardPricingResult(station_prices, station_finance, community_satisfaction, demand_release, summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/capacity_hard_pricing")
    parser.add_argument("--scenarios", default="baseline,elderly_growth_8pct,transition_up,fixed_cost_up_20pct,budget_140")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    all_prices = []
    all_finance = []
    all_satisfaction = []
    all_release = []
    for scenario in [s.strip() for s in args.scenarios.split(",") if s.strip()]:
        print(f"[capacity-hard-p3] scenario start: {scenario}", flush=True)
        result = run_scenario(scenario, output_dir)
        summaries.append(result.scenario_summary)
        all_prices.append(result.station_prices)
        all_finance.append(result.station_finance)
        all_satisfaction.append(result.community_satisfaction)
        all_release.append(result.demand_release)
        print(f"[capacity-hard-p3] scenario finished: {scenario}", flush=True)

    pd.DataFrame(summaries).to_csv(output_dir / "capacity_hard_problem4_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_prices, ignore_index=True).to_csv(output_dir / "capacity_hard_all_station_prices.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_finance, ignore_index=True).to_csv(output_dir / "capacity_hard_all_station_finance.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_satisfaction, ignore_index=True).to_csv(output_dir / "capacity_hard_all_community_satisfaction.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_release, ignore_index=True).to_csv(output_dir / "capacity_hard_all_demand_release.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(output_dir / "capacity_hard_results_summary.xlsx") as writer:
        pd.DataFrame(summaries).to_excel(writer, sheet_name="problem4_summary", index=False)
        pd.concat(all_prices, ignore_index=True).to_excel(writer, sheet_name="problem3_prices", index=False)
        pd.concat(all_finance, ignore_index=True).to_excel(writer, sheet_name="problem3_finance", index=False)
        pd.concat(all_satisfaction, ignore_index=True).to_excel(writer, sheet_name="community_satisfaction", index=False)
        pd.concat(all_release, ignore_index=True).to_excel(writer, sheet_name="demand_release", index=False)


if __name__ == "__main__":
    main()
