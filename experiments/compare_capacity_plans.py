from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import deep_update, load_config
from src.data.loaders import RawData, load_raw_data
from src.models.problem1_markov_demand import MarkovDemandModel
from src.models.problem2_cmclp_bilevel import LocationCapacityOptimizer
from src.utils.metrics import distance_satisfaction, response_satisfaction


@dataclass
class CapacityEvaluation:
    label: str
    plan: dict[str, dict[str, float | str]]
    metrics: dict[str, float | int | str]
    assignment: pd.DataFrame
    station_status: pd.DataFrame
    station_profit: pd.DataFrame


def clone_raw(raw: RawData) -> RawData:
    return RawData(
        communities=raw.communities.copy(deep=True),
        transitions=dict(raw.transitions),
        demand_freq=raw.demand_freq.copy(deep=True),
        service_cost=raw.service_cost.copy(deep=True),
        consumption_caps=dict(raw.consumption_caps),
        station_cost=raw.station_cost.copy(deep=True),
        distance=raw.distance.copy(deep=True),
    )


def apply_scenario(raw: RawData, cfg: dict[str, Any], scenario: str) -> tuple[RawData, dict[str, Any]]:
    if scenario == "baseline":
        return raw, cfg
    sc = cfg["problem4"]["scenarios"][scenario]
    raw_s = clone_raw(raw)
    cfg_s = json.loads(json.dumps(cfg, ensure_ascii=False))
    if "problem1" in sc:
        deep_update(cfg_s, {"problem1": sc["problem1"]})
    if "problem2" in sc:
        deep_update(cfg_s, {"problem2": sc["problem2"]})
    if "transitions" in sc:
        for k, v in sc["transitions"].items():
            raw_s.transitions[k] = float(v)
    if "cost_multiplier" in sc and "fixed_daily_cost" in sc["cost_multiplier"]:
        raw_s.station_cost["fixed_daily_cost"] *= float(sc["cost_multiplier"]["fixed_daily_cost"])
    return raw_s, cfg_s


def make_plan(scale_map: dict[str, dict[str, Any]], spec: dict[str, str]) -> dict[str, dict[str, float | str]]:
    return {
        loc: {
            "scale": scale,
            "capacity": float(scale_map[scale]["daily_capacity"]),
            "build_cost_wan": float(scale_map[scale]["build_cost_wan"]),
            "fixed_daily_cost": float(scale_map[scale]["fixed_daily_cost"]),
        }
        for loc, scale in spec.items()
    }


def enumerate_plans(communities: list[str], scale_records: list[dict[str, Any]], budget_wan: float):
    plan_id = 0
    for k in range(1, len(communities) + 1):
        for locs in combinations(communities, k):
            for scales in product(scale_records, repeat=k):
                cost = sum(float(s["build_cost_wan"]) for s in scales)
                if cost > budget_wan:
                    continue
                plan_id += 1
                yield plan_id, {
                    loc: {
                        "scale": s["scale"],
                        "capacity": float(s["daily_capacity"]),
                        "build_cost_wan": float(s["build_cost_wan"]),
                        "fixed_daily_cost": float(s["fixed_daily_cost"]),
                    }
                    for loc, s in zip(locs, scales)
                }


def hard_capacity_allocate(
    raw: RawData,
    cfg: dict[str, Any],
    p1: Any,
    plan: dict[str, dict[str, float | str]],
    label: str,
    min_community_fulfillment: float = 0.0,
) -> CapacityEvaluation:
    radius = float(cfg["problem2"]["service_radius_m"])
    days_per_month = float(cfg["problem2"]["days_per_month"])
    days_per_year = float(cfg["problem2"]["days_per_year"])
    communities = list(raw.communities["community"])
    stations = list(plan.keys())

    monthly_by_service = p1.budgeted_monthly_demand.groupby(["community", "service"], as_index=False)["monthly_demand"].sum()
    daily_demand = (monthly_by_service.groupby("community")["monthly_demand"].sum() / days_per_month).to_dict()
    total_daily_demand = float(sum(daily_demand.values()))
    elderly = p1.year5_population.set_index("community")["elderly_total"].to_dict()
    total_elderly = float(sum(elderly.values()))

    pairs: list[tuple[str, str]] = []
    c_obj: list[float] = []
    dist_scores: dict[tuple[str, str], float] = {}
    distances: dict[tuple[str, str], float] = {}
    for i in communities:
        for j in stations:
            d = float(raw.distance.loc[i, j])
            if d <= radius:
                pairs.append((i, j))
                ds = distance_satisfaction(d)
                dist_scores[(i, j)] = ds
                distances[(i, j)] = d
                c_obj.append(-ds)

    if not pairs:
        raise RuntimeError(f"No reachable community-station pairs for {label}")

    n = len(pairs)
    a_ub = []
    b_ub = []
    for i in communities:
        row = [0.0] * n
        for idx, (ci, _) in enumerate(pairs):
            if ci == i:
                row[idx] = 1.0
        a_ub.append(row)
        b_ub.append(float(daily_demand[i]))
    for j in stations:
        row = [0.0] * n
        for idx, (_, sj) in enumerate(pairs):
            if sj == j:
                row[idx] = 1.0
        a_ub.append(row)
        b_ub.append(float(plan[j]["capacity"]))
    min_community_fulfillment = float(min_community_fulfillment)
    if min_community_fulfillment > 0:
        for i in communities:
            row = [0.0] * n
            for idx, (ci, _) in enumerate(pairs):
                if ci == i:
                    row[idx] = -1.0
            a_ub.append(row)
            b_ub.append(-min_community_fulfillment * float(daily_demand[i]))

    res = linprog(c_obj, A_ub=np.asarray(a_ub), b_ub=np.asarray(b_ub), bounds=[(0, None)] * n, method="highs")
    if not res.success:
        raise RuntimeError(f"Capacity allocation failed for {label}: {res.message}")

    alloc_rows = []
    x = np.asarray(res.x, dtype=float)
    for val, (i, j) in zip(x, pairs):
        if val <= 1e-7:
            continue
        alloc_rows.append(
            {
                "community": i,
                "station": j,
                "distance_m": distances[(i, j)],
                "distance_score": dist_scores[(i, j)],
                "allocated_daily_demand": float(val),
            }
        )
    assignment = pd.DataFrame(alloc_rows)
    if assignment.empty:
        assignment = pd.DataFrame(columns=["community", "station", "distance_m", "distance_score", "allocated_daily_demand"])

    station_load = assignment.groupby("station")["allocated_daily_demand"].sum().to_dict()
    station_response = {
        j: response_satisfaction(float(station_load.get(j, 0.0)) / float(plan[j]["capacity"]))
        for j in stations
    }
    assignment["response_score"] = assignment["station"].map(station_response)
    assignment["price_score"] = 1.0
    assignment["satisfaction"] = 0.2 * assignment["distance_score"] + 0.3 * assignment["response_score"] + 0.5
    assignment["effective_daily_demand"] = assignment["allocated_daily_demand"] * assignment["satisfaction"]

    served_by_community = assignment.groupby("community")["allocated_daily_demand"].sum().to_dict()
    effective_by_community = assignment.groupby("community")["effective_daily_demand"].sum().to_dict()
    best_station = (
        assignment.sort_values(["community", "allocated_daily_demand", "satisfaction"], ascending=[True, False, False])
        .groupby("community")
        .first()
        .reset_index()
        if not assignment.empty
        else pd.DataFrame()
    )
    covered_spatial = {
        i: any(float(raw.distance.loc[i, j]) <= radius for j in stations)
        for i in communities
    }
    spatial_covered_elderly = sum(float(elderly[i]) for i in communities if covered_spatial[i])
    served_elderly_proxy = sum(float(elderly[i]) * min(1.0, served_by_community.get(i, 0.0) / float(daily_demand[i])) for i in communities)

    community_rows = []
    for i in communities:
        served = float(served_by_community.get(i, 0.0))
        eff = float(effective_by_community.get(i, 0.0))
        sub = assignment[assignment["community"] == i]
        weighted_sat = eff / served if served > 1e-9 else 0.0
        actual_stations = ",".join(sub["station"].astype(str).tolist()) if len(sub) else ""
        community_rows.append(
            {
                "community": i,
                "daily_demand": float(daily_demand[i]),
                "served_daily_demand": served,
                "unserved_daily_demand": max(0.0, float(daily_demand[i]) - served),
                "capacity_fulfillment_rate": served / float(daily_demand[i]) if daily_demand[i] > 0 else 0.0,
                "effective_daily_demand": eff,
                "served_weighted_satisfaction": weighted_sat,
                "actual_stations": actual_stations,
                "spatially_covered": int(covered_spatial[i]),
            }
        )
    community_assignment = pd.DataFrame(community_rows)

    station_rows = []
    for j in stations:
        load = float(station_load.get(j, 0.0))
        sub = assignment[assignment["station"] == j]
        station_rows.append(
            {
                "station": j,
                "scale": plan[j]["scale"],
                "capacity": float(plan[j]["capacity"]),
                "build_cost_wan": float(plan[j]["build_cost_wan"]),
                "fixed_daily_cost": float(plan[j]["fixed_daily_cost"]),
                "served_daily_demand": load,
                "utilization": load / float(plan[j]["capacity"]) if float(plan[j]["capacity"]) > 0 else 0.0,
                "response_score": station_response[j],
                "assigned_communities": ",".join(sorted(sub["community"].unique().tolist())) if len(sub) else "",
            }
        )
    station_status = pd.DataFrame(station_rows)

    service_cost = raw.service_cost.set_index("service")
    monthly = monthly_by_service.copy()
    community_served_rate = {
        r["community"]: r["capacity_fulfillment_rate"]
        for _, r in community_assignment.iterrows()
    }
    monthly["served_monthly_demand"] = monthly.apply(
        lambda r: float(r["monthly_demand"]) * float(community_served_rate.get(r["community"], 0.0)),
        axis=1,
    )
    # Attribute each community to its largest actual station for a simple no-subsidy profit table.
    dominant = {}
    if not assignment.empty:
        for i, sub in assignment.groupby("community"):
            top = sub.sort_values(["allocated_daily_demand", "satisfaction"], ascending=[False, False]).iloc[0]
            dominant[i] = top["station"]
    monthly["station"] = monthly["community"].map(dominant)
    monthly = monthly.dropna(subset=["station"])
    monthly["base_price"] = monthly["service"].map(service_cost["base_price"])
    monthly["direct_cost"] = monthly["service"].map(service_cost["direct_cost"])
    monthly["annual_revenue"] = monthly["served_monthly_demand"] * monthly["base_price"] * 12
    monthly["annual_direct_cost"] = monthly["served_monthly_demand"] * monthly["direct_cost"] * 12
    station_profit = monthly.groupby("station", as_index=False)[["annual_revenue", "annual_direct_cost"]].sum()
    station_profit["fixed_management_cost"] = station_profit["station"].map({j: float(plan[j]["fixed_daily_cost"]) * days_per_year for j in stations})
    station_profit["annual_depreciation"] = station_profit["station"].map({j: float(plan[j]["build_cost_wan"]) * 10000 / 20 for j in stations})
    station_profit["annual_profit_without_subsidy"] = (
        station_profit["annual_revenue"]
        - station_profit["annual_direct_cost"]
        - station_profit["fixed_management_cost"]
        - station_profit["annual_depreciation"]
    )
    for j in stations:
        if j not in set(station_profit["station"]):
            fixed = float(plan[j]["fixed_daily_cost"]) * days_per_year
            dep = float(plan[j]["build_cost_wan"]) * 10000 / 20
            station_profit = pd.concat(
                [
                    station_profit,
                    pd.DataFrame(
                        [
                            {
                                "station": j,
                                "annual_revenue": 0.0,
                                "annual_direct_cost": 0.0,
                                "fixed_management_cost": fixed,
                                "annual_depreciation": dep,
                                "annual_profit_without_subsidy": -(fixed + dep),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    total_served = float(community_assignment["served_daily_demand"].sum())
    total_effective = float(community_assignment["effective_daily_demand"].sum())
    metrics = {
        "spatial_coverage_rate": spatial_covered_elderly / total_elderly if total_elderly > 0 else 0.0,
        "capacity_fulfillment_rate": total_served / total_daily_demand if total_daily_demand > 0 else 0.0,
        "effective_service_rate": total_effective / total_daily_demand if total_daily_demand > 0 else 0.0,
        "served_weighted_satisfaction": total_effective / total_served if total_served > 0 else 0.0,
        "served_elderly_proxy_rate": served_elderly_proxy / total_elderly if total_elderly > 0 else 0.0,
        "min_community_fulfillment_rate": float(community_assignment["capacity_fulfillment_rate"].min()) if len(community_assignment) else 0.0,
        "total_daily_demand": total_daily_demand,
        "served_daily_demand": total_served,
        "unserved_daily_demand": max(0.0, total_daily_demand - total_served),
        "total_effective_daily_demand": total_effective,
        "total_capacity": float(sum(float(p["capacity"]) for p in plan.values())),
        "total_build_cost_wan": float(sum(float(p["build_cost_wan"]) for p in plan.values())),
        "station_count": len(stations),
        "max_utilization": float(station_status["utilization"].max()) if len(station_status) else 0.0,
        "annual_profit_without_subsidy": float(station_profit["annual_profit_without_subsidy"].sum()),
        "locations": ",".join(stations),
        "scales": ",".join(str(plan[j]["scale"]) for j in stations),
    }
    # Store community-level actual allocation in assignment output, not just pair rows.
    assignment_out = community_assignment.merge(
        best_station[["community", "station", "distance_m", "distance_score", "satisfaction"]].rename(
            columns={
                "station": "dominant_station",
                "distance_m": "dominant_distance_m",
                "distance_score": "dominant_distance_score",
                "satisfaction": "dominant_satisfaction",
            }
        )
        if len(best_station)
        else pd.DataFrame(columns=["community", "dominant_station", "dominant_distance_m", "dominant_distance_score", "dominant_satisfaction"]),
        on="community",
        how="left",
    )
    return CapacityEvaluation(label, plan, metrics, assignment_out, station_status, station_profit)


def rank_key(ev: CapacityEvaluation):
    m = ev.metrics
    return (
        round(float(m["spatial_coverage_rate"]), 9),
        round(float(m["capacity_fulfillment_rate"]), 9),
        round(float(m["min_community_fulfillment_rate"]), 9),
        float(m["effective_service_rate"]),
        float(m["served_weighted_satisfaction"]),
        -float(m["total_build_cost_wan"]),
        -int(m["station_count"]),
    )


def add_rank_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_spatial_rank"] = out["spatial_coverage_rate"].round(9)
    out["_capacity_rank"] = out["capacity_fulfillment_rate"].round(9)
    out["_min_fulfillment_rank"] = out["min_community_fulfillment_rate"].round(9)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--scenario", default="baseline")
    parser.add_argument("--output-dir", default="outputs/capacity_optimized")
    parser.add_argument("--enumerate", action="store_true")
    parser.add_argument("--min-community-fulfillment", type=float, default=0.0)
    args = parser.parse_args()

    cfg0 = load_config(args.config)
    raw0 = load_raw_data(cfg0)
    raw, cfg = apply_scenario(raw0, cfg0, args.scenario)
    p1 = MarkovDemandModel(raw, cfg).run()
    helper = LocationCapacityOptimizer(raw, cfg, p1)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_specs = {
        "current_CDG": {"C": "小型", "D": "大型", "G": "大型"},
        "online_BDFJ": {"B": "大型", "D": "小型", "F": "中型", "J": "小型"},
        "online_budget140_BFGJ": {"B": "小型", "F": "中型", "G": "大型", "J": "大型"},
    }
    evaluations: list[CapacityEvaluation] = []
    for label, spec in candidate_specs.items():
        cost = sum(float(helper.scale_map[s]["build_cost_wan"]) for s in spec.values())
        if cost <= float(cfg["problem2"]["budget_wan"]):
            try:
                ev = hard_capacity_allocate(
                    raw,
                    cfg,
                    p1,
                    make_plan(helper.scale_map, spec),
                    label,
                    min_community_fulfillment=args.min_community_fulfillment,
                )
            except RuntimeError as exc:
                print(f"[skip] {label}: {exc}", flush=True)
                continue
            evaluations.append(ev)
            ev.assignment.to_csv(out_dir / f"{args.scenario}_{label}_assignment.csv", index=False, encoding="utf-8-sig")
            ev.station_status.to_csv(out_dir / f"{args.scenario}_{label}_station_status.csv", index=False, encoding="utf-8-sig")
            ev.station_profit.to_csv(out_dir / f"{args.scenario}_{label}_profit_without_subsidy.csv", index=False, encoding="utf-8-sig")

    if args.enumerate:
        best: CapacityEvaluation | None = None
        rows = []
        communities = list(raw.communities["community"])
        scale_records = raw.station_cost.to_dict("records")
        for pid, plan in enumerate_plans(communities, scale_records, float(cfg["problem2"]["budget_wan"])):
            # Skip plans with no spatial full coverage only after recording enough objective info.
            try:
                ev = hard_capacity_allocate(
                    raw,
                    cfg,
                    p1,
                    plan,
                    f"enum_{pid}",
                    min_community_fulfillment=args.min_community_fulfillment,
                )
            except RuntimeError:
                continue
            row = {"plan_id": pid, **ev.metrics}
            rows.append(row)
            if best is None or rank_key(ev) > rank_key(best):
                best = ev
        summary = add_rank_columns(pd.DataFrame(rows)).sort_values(
            [
                "_spatial_rank",
                "_capacity_rank",
                "_min_fulfillment_rank",
                "effective_service_rate",
                "served_weighted_satisfaction",
                "total_build_cost_wan",
            ],
            ascending=[False, False, False, False, False, True],
        ).drop(columns=["_spatial_rank", "_capacity_rank", "_min_fulfillment_rank"])
        summary.to_csv(out_dir / f"{args.scenario}_enumeration_summary.csv", index=False, encoding="utf-8-sig")
        if best is not None:
            best.label = "enumerated_best"
            evaluations.append(best)
            best.assignment.to_csv(out_dir / f"{args.scenario}_enumerated_best_assignment.csv", index=False, encoding="utf-8-sig")
            best.station_status.to_csv(out_dir / f"{args.scenario}_enumerated_best_station_status.csv", index=False, encoding="utf-8-sig")
            best.station_profit.to_csv(out_dir / f"{args.scenario}_enumerated_best_profit_without_subsidy.csv", index=False, encoding="utf-8-sig")

    comparison = add_rank_columns(pd.DataFrame([{"label": ev.label, **ev.metrics} for ev in evaluations])).sort_values(
        [
            "_spatial_rank",
            "_capacity_rank",
            "_min_fulfillment_rank",
            "effective_service_rate",
            "served_weighted_satisfaction",
            "total_build_cost_wan",
        ],
        ascending=[False, False, False, False, False, True],
    ).drop(columns=["_spatial_rank", "_capacity_rank", "_min_fulfillment_rank"])
    comparison.to_csv(out_dir / f"{args.scenario}_plan_comparison.csv", index=False, encoding="utf-8-sig")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
