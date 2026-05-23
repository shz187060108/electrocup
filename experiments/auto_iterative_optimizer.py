from __future__ import annotations

import argparse
import json
import math
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import deep_update, ensure_dirs, load_config
from src.data.loaders import RawData, load_raw_data
from src.models.problem1_markov_demand import MarkovDemandModel
from src.models.problem2_cmclp_bilevel import LocationCapacityOptimizer
from src.models.problem3_pricing_subsidy import PricingSubsidyOptimizer
from src.utils.io import save_dataframe, save_json


FULL_COVERAGE_TOL = 1e-6


@dataclass
class IterationRecord:
    iteration: int
    cycle: int
    scenario: str
    price_profile: str
    policy: str
    plan_id: int
    locations: str
    scales: str
    station_count: int
    total_build_cost_wan: float
    coverage_rate: float
    average_satisfaction: float
    min_satisfaction: float
    overload: float
    max_utilization: float
    threshold_drop_loss: float
    annual_profit_yuan_problem2: float
    annual_subsidy_yuan_problem3: float
    avg_profit_rate_problem3: float
    min_profit_rate_problem3: float
    pricing_feasible: int
    weighted_price_satisfaction_problem3: float
    elapsed_seconds: float


def raw_clone(raw: RawData) -> RawData:
    return RawData(
        communities=raw.communities.copy(deep=True),
        transitions=dict(raw.transitions),
        demand_freq=raw.demand_freq.copy(deep=True),
        service_cost=raw.service_cost.copy(deep=True),
        consumption_caps=dict(raw.consumption_caps),
        station_cost=raw.station_cost.copy(deep=True),
        distance=raw.distance.copy(deep=True),
    )


def apply_raw_scenario(raw: RawData, scenario_cfg: dict[str, Any]) -> RawData:
    out = raw_clone(raw)
    for key, value in scenario_cfg.get("transitions", {}).items():
        out.transitions[key] = float(value)
    mult = scenario_cfg.get("cost_multiplier", {})
    if "fixed_daily_cost" in mult:
        out.station_cost["fixed_daily_cost"] *= float(mult["fixed_daily_cost"])
    return out


def scenario_definitions(cfg: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    scenarios = [("baseline", {})]
    scenarios.extend((name, patch) for name, patch in cfg["problem4"]["scenarios"].items() if name != "baseline")
    return scenarios


def price_profiles() -> list[tuple[str, list[float]]]:
    return [
        ("base_grid", [0.60, 0.70, 0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]),
        ("low_fine_grid", [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]),
        ("profit_edge_grid", [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 1.00, 1.02, 1.05, 1.10]),
    ]


def select_policy_row(summary: pd.DataFrame, policy: str) -> pd.Series:
    df = summary.copy()
    full = df[df["coverage_rate"] >= 1.0 - FULL_COVERAGE_TOL]
    if policy == "legacy_score":
        return df.sort_values("score", ascending=False).iloc[0]
    if policy == "coverage_capacity":
        pool = full if len(full) else df[df["coverage_rate"] == df["coverage_rate"].max()]
        return pool.sort_values(
            ["overload", "average_satisfaction", "min_satisfaction", "threshold_drop_loss", "annual_profit_yuan"],
            ascending=[True, False, False, True, False],
        ).iloc[0]
    if policy == "coverage_satisfaction_guarded":
        pool = full if len(full) else df[df["coverage_rate"] == df["coverage_rate"].max()]
        min_overload = float(pool["overload"].min())
        guarded = pool[pool["overload"] <= min_overload + 0.03]
        return guarded.sort_values(
            ["average_satisfaction", "min_satisfaction", "overload", "threshold_drop_loss", "annual_profit_yuan"],
            ascending=[False, False, True, True, False],
        ).iloc[0]
    if policy == "minimax_loss":
        pool = full if len(full) else df
        return pool.sort_values(
            ["threshold_drop_loss", "drop_loss", "overload", "average_satisfaction", "annual_profit_yuan"],
            ascending=[True, True, True, False, False],
        ).iloc[0]
    if policy == "profit_with_service_floor":
        pool = full if len(full) else df
        guarded = pool[(pool["average_satisfaction"] >= 0.85) & (pool["overload"] <= 0.12)]
        if len(guarded) == 0:
            guarded = pool
        return guarded.sort_values(
            ["annual_profit_yuan", "overload", "average_satisfaction"],
            ascending=[False, True, False],
        ).iloc[0]
    raise ValueError(f"Unknown policy: {policy}")


def plan_by_id(optimizer: LocationCapacityOptimizer, plan_id: int) -> dict[str, dict[str, Any]]:
    for pid, plan in optimizer.enumerate_plans():
        if int(pid) == int(plan_id):
            return plan
    raise KeyError(f"Plan id not found: {plan_id}")


def persist_selected_outputs(
    out_dir: Path,
    label: str,
    p1: Any,
    p2_eval: Any,
    p3: Any,
    full_summary: pd.DataFrame | None = None,
) -> None:
    prefix = out_dir / label
    save_dataframe(p1.year5_population, prefix.with_name(prefix.name + "_problem1_year5_population.csv"))
    save_dataframe(p2_eval.fixed_point.assignment_table, prefix.with_name(prefix.name + "_problem2_assignment.csv"))
    save_dataframe(p2_eval.fixed_point.station_table, prefix.with_name(prefix.name + "_problem2_station_status.csv"))
    save_json(
        {k: v for k, v in p2_eval.metrics.items() if k != "station_profit_table"},
        prefix.with_name(prefix.name + "_problem2_metrics.json"),
    )
    save_dataframe(p2_eval.metrics["station_profit_table"], prefix.with_name(prefix.name + "_problem2_station_profit.csv"))
    save_dataframe(p3.station_prices, prefix.with_name(prefix.name + "_problem3_station_prices.csv"))
    save_dataframe(p3.station_finance, prefix.with_name(prefix.name + "_problem3_station_finance.csv"))
    save_dataframe(p3.community_satisfaction, prefix.with_name(prefix.name + "_problem3_community_satisfaction.csv"))
    save_dataframe(p3.demand_release, prefix.with_name(prefix.name + "_problem3_latent_demand_release.csv"))
    if full_summary is not None:
        save_dataframe(full_summary, prefix.with_name(prefix.name + "_problem2_full_summary.csv"))


def run_iteration(
    iteration: int,
    cycle: int,
    scenario_name: str,
    scenario_patch: dict[str, Any],
    price_profile_name: str,
    multipliers: list[float],
    cfg_base: dict[str, Any],
    raw_base: RawData,
    out_dir: Path,
    started_at: float,
    policies: list[str],
) -> list[IterationRecord]:
    cfg = deepcopy(cfg_base)
    raw = apply_raw_scenario(raw_base, scenario_patch)
    if "problem1" in scenario_patch:
        deep_update(cfg, {"problem1": scenario_patch["problem1"]})
    if "problem2" in scenario_patch:
        deep_update(cfg, {"problem2": scenario_patch["problem2"]})
    cfg["problem2"]["enumeration"]["show_progress"] = False
    cfg["problem3"]["candidate_price_multipliers"] = multipliers
    cfg["problem3"]["pricing_search"]["max_combinations_per_station"] = None

    p1 = MarkovDemandModel(raw, cfg).run()
    optimizer = LocationCapacityOptimizer(raw, cfg, p1)
    _, summary = optimizer.run()

    selected: dict[str, int] = {}
    for policy in policies:
        row = select_policy_row(summary, policy)
        selected[policy] = int(row["plan_id"])

    records: list[IterationRecord] = []
    evaluated_plans: dict[int, Any] = {}
    evaluated_prices: dict[int, Any] = {}
    for policy, plan_id in selected.items():
        if plan_id not in evaluated_plans:
            plan = plan_by_id(optimizer, plan_id)
            evaluated_plans[plan_id] = optimizer.evaluate_plan(plan_id, plan)
            evaluated_prices[plan_id] = PricingSubsidyOptimizer(raw, cfg, p1, evaluated_plans[plan_id]).run()
        p2_eval = evaluated_plans[plan_id]
        p3 = evaluated_prices[plan_id]
        label = f"iter{iteration:04d}_{scenario_name}_{price_profile_name}_{policy}"
        persist_selected_outputs(
            out_dir,
            label,
            p1,
            p2_eval,
            p3,
            full_summary=summary if policy == "coverage_capacity" else None,
        )

        finance = p3.station_finance
        records.append(
            IterationRecord(
                iteration=iteration,
                cycle=cycle,
                scenario=scenario_name,
                price_profile=price_profile_name,
                policy=policy,
                plan_id=plan_id,
                locations=",".join(p2_eval.plan.keys()),
                scales=",".join([p2_eval.plan[j]["scale"] for j in p2_eval.plan.keys()]),
                station_count=int(p2_eval.metrics["station_count"]),
                total_build_cost_wan=float(p2_eval.metrics["total_build_cost_wan"]),
                coverage_rate=float(p2_eval.metrics["coverage_rate"]),
                average_satisfaction=float(p2_eval.metrics["average_satisfaction"]),
                min_satisfaction=float(p2_eval.metrics["min_satisfaction"]),
                overload=float(p2_eval.metrics["overload"]),
                max_utilization=float(p2_eval.metrics["max_utilization"]),
                threshold_drop_loss=float(p2_eval.metrics["threshold_drop_loss"]),
                annual_profit_yuan_problem2=float(p2_eval.metrics["annual_profit_yuan"]),
                annual_subsidy_yuan_problem3=float(finance["annual_subsidy"].sum()),
                avg_profit_rate_problem3=float(finance["profit_rate"].mean()),
                min_profit_rate_problem3=float(finance["profit_rate"].min()),
                pricing_feasible=int(finance["pricing_feasible"].min()),
                weighted_price_satisfaction_problem3=float(finance["weighted_price_satisfaction"].mean()),
                elapsed_seconds=float(time.time() - started_at),
            )
        )
    return records


def append_records(path: Path, records: list[IterationRecord]) -> None:
    df = pd.DataFrame([asdict(r) for r in records])
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")


def write_leaderboard(path: Path, records_path: Path) -> None:
    if not records_path.exists():
        return
    df = pd.read_csv(records_path)
    full = df[df["coverage_rate"] >= 1.0 - FULL_COVERAGE_TOL]
    service_pool = full if len(full) else df[df["coverage_rate"] == df["coverage_rate"].max()]
    service_best = service_pool.sort_values(
        ["overload", "average_satisfaction", "min_satisfaction", "weighted_price_satisfaction_problem3"],
        ascending=[True, False, False, False],
    ).head(20)
    satisfaction_best = service_pool.sort_values(
        ["average_satisfaction", "overload", "min_satisfaction", "weighted_price_satisfaction_problem3"],
        ascending=[False, True, False, False],
    ).head(20)
    profit_best = service_pool.sort_values(
        ["pricing_feasible", "annual_profit_yuan_problem2", "overload"],
        ascending=[False, False, True],
    ).head(20)
    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, sheet_name="all_iterations", index=False)
        service_best.to_excel(writer, sheet_name="service_recommended", index=False)
        satisfaction_best.to_excel(writer, sheet_name="satisfaction_frontier", index=False)
        profit_best.to_excel(writer, sheet_name="profit_frontier", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="outputs/auto_iter")
    parser.add_argument("--duration-hours", type=float, default=8.0)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--single-pass", action="store_true")
    parser.add_argument("--price-profile-filter", default=None)
    parser.add_argument(
        "--policies",
        default="legacy_score,coverage_capacity,coverage_satisfaction_guarded,minimax_loss,profit_with_service_floor",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    raw = load_raw_data(cfg)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "iteration_records.csv"
    leaderboard_path = out_dir / "leaderboard.xlsx"
    manifest_path = out_dir / "manifest.json"

    scenarios = scenario_definitions(cfg)
    profiles = price_profiles()
    if args.price_profile_filter:
        allowed_profiles = {p.strip() for p in args.price_profile_filter.split(",") if p.strip()}
        profiles = [p for p in profiles if p[0] in allowed_profiles]
        if not profiles:
            raise ValueError(f"No price profiles matched: {args.price_profile_filter}")
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    if not policies:
        raise ValueError("At least one policy is required.")
    started_at = time.time()
    deadline = started_at + max(0.0, args.duration_hours) * 3600.0
    iteration = 0
    cycle = 0
    manifest = {
        "started_at_epoch": started_at,
        "duration_hours": args.duration_hours,
        "scenarios": [s[0] for s in scenarios],
        "price_profiles": [p[0] for p in profiles],
        "policies": policies,
        "objective_note": (
            "Problem 2 is exhaustively enumerated per scenario. Leaderboards use service-first "
            "multi-objective ranking because the legacy scalar score over-rewards low overload "
            "even when coverage is poor."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    while True:
        cycle += 1
        for scenario_name, scenario_patch in scenarios:
            for price_profile_name, multipliers in profiles:
                iteration += 1
                print(
                    f"[auto-iter] iteration={iteration} cycle={cycle} "
                    f"scenario={scenario_name} price_profile={price_profile_name}",
                    flush=True,
                )
                records = run_iteration(
                    iteration,
                    cycle,
                    scenario_name,
                    scenario_patch,
                    price_profile_name,
                    multipliers,
                    cfg,
                    raw,
                    out_dir,
                    started_at,
                    policies,
                )
                append_records(records_path, records)
                write_leaderboard(leaderboard_path, records_path)

                elapsed = time.time() - started_at
                print(f"[auto-iter] iteration={iteration} finished elapsed_seconds={elapsed:.1f}", flush=True)
                if args.single_pass:
                    return
                if args.max_iterations is not None and iteration >= args.max_iterations:
                    return
                if time.time() >= deadline and iteration > 0:
                    return
        if args.duration_hours <= 0 and args.max_iterations is None:
            return


if __name__ == "__main__":
    main()
