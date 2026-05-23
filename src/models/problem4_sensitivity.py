from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import pandas as pd

from src.config import deep_update
from src.data.loaders import RawData
from src.models.problem1_markov_demand import MarkovDemandModel
from src.models.problem2_cmclp_bilevel import LocationCapacityOptimizer
from src.models.problem3_pricing_subsidy import PricingSubsidyOptimizer


@dataclass
class SensitivityResult:
    scenario_summary: pd.DataFrame
    scenario_plans: dict


class SensitivityAnalyzer:
    """问题四：灵敏度分析与方案比较。

    完整版求解逻辑：每个情景均重新运行问题一、问题二和问题三。
    同时将基准方案放到扰动情景下重新评价，计算后悔值和方案稳定性。
    """

    def __init__(self, raw, cfg: dict, baseline_p2=None):
        self.raw = raw
        self.cfg = cfg
        self.baseline_p2 = baseline_p2

    def _apply_scenario_to_raw(self, raw, scenario_cfg: dict):
        r = RawData(
            communities=raw.communities.copy(deep=True),
            transitions=dict(raw.transitions),
            demand_freq=raw.demand_freq.copy(deep=True),
            service_cost=raw.service_cost.copy(deep=True),
            consumption_caps=dict(raw.consumption_caps),
            station_cost=raw.station_cost.copy(deep=True),
            distance=raw.distance.copy(deep=True),
        )
        if 'transitions' in scenario_cfg:
            for k, v in scenario_cfg['transitions'].items():
                r.transitions[k] = float(v)
        if 'cost_multiplier' in scenario_cfg:
            mult = scenario_cfg['cost_multiplier']
            if 'fixed_daily_cost' in mult:
                r.station_cost['fixed_daily_cost'] *= float(mult['fixed_daily_cost'])
        return r

    @staticmethod
    def _station_stability(base_locations: set, new_locations: set) -> float:
        union = base_locations | new_locations
        return len(base_locations & new_locations) / len(union) if union else 1.0

    @staticmethod
    def _scale_adjustment_rate(base_plan: dict, new_plan: dict) -> float:
        if not base_plan:
            return 0.0
        changed = 0
        for loc, info in base_plan.items():
            if loc not in new_plan or new_plan[loc]['scale'] != info['scale']:
                changed += 1
        return changed / len(base_plan)

    @staticmethod
    def _safe_robustness_index(baseline_score: float | None, optimal_score: float | None) -> float | None:
        if baseline_score is None or optimal_score is None:
            return None
        denom = abs(optimal_score)
        if denom < 1e-12:
            return 1.0 if abs(baseline_score - optimal_score) < 1e-12 else 0.0
        return 1.0 - max(0.0, optimal_score - baseline_score) / denom

    def run(self) -> SensitivityResult:
        base_plan = self.baseline_p2.plan if self.baseline_p2 is not None else {}
        base_locations = set(base_plan.keys())
        rows, plans = [], {}
        scenarios = self.cfg['problem4']['scenarios']

        for name, sc in scenarios.items():
            print(f'[Problem4] scenario start: {name}', flush=True)
            cfg_s = deepcopy(self.cfg)
            raw_s = self._apply_scenario_to_raw(self.raw, sc)
            if 'problem1' in sc:
                deep_update(cfg_s, {'problem1': sc['problem1']})
            if 'problem2' in sc:
                deep_update(cfg_s, {'problem2': sc['problem2']})

            p1_s = MarkovDemandModel(raw_s, cfg_s).run()
            optimizer_s = LocationCapacityOptimizer(raw_s, cfg_s, p1_s)
            p2_opt, p2_summary = optimizer_s.run()
            p3_opt = PricingSubsidyOptimizer(raw_s, cfg_s, p1_s, p2_opt).run()

            baseline_eval_s = None
            baseline_p3_s = None
            if self.baseline_p2 is not None:
                try:
                    baseline_eval_s = optimizer_s.evaluate_plan(0, base_plan)
                    baseline_p3_s = PricingSubsidyOptimizer(raw_s, cfg_s, p1_s, baseline_eval_s).run()
                except Exception as exc:
                    print(f'[Problem4] baseline plan evaluation failed under {name}: {exc}', flush=True)

            locations = set(p2_opt.plan.keys())
            stability = self._station_stability(base_locations, locations) if base_locations else 1.0
            scale_adjust = self._scale_adjustment_rate(base_plan, p2_opt.plan) if base_plan else 0.0
            baseline_score_s = baseline_eval_s.metrics['score'] if baseline_eval_s is not None else None
            optimal_score_s = p2_opt.metrics['score']
            regret = max(0.0, optimal_score_s - baseline_score_s) if baseline_score_s is not None else None
            robustness_index = self._safe_robustness_index(baseline_score_s, optimal_score_s)

            rows.append({
                'scenario': name,
                'description': sc.get('description', name),
                'locations': ','.join(p2_opt.plan.keys()),
                'scales': ','.join([p2_opt.plan[j]['scale'] for j in p2_opt.plan.keys()]),
                'station_count': p2_opt.metrics['station_count'],
                'total_build_cost_wan': p2_opt.metrics['total_build_cost_wan'],
                'coverage_rate': p2_opt.metrics['coverage_rate'],
                'average_satisfaction': p2_opt.metrics['average_satisfaction'],
                'min_satisfaction': p2_opt.metrics['min_satisfaction'],
                'drop_loss': p2_opt.metrics['drop_loss'],
                'threshold_drop_loss': p2_opt.metrics['threshold_drop_loss'],
                'distance_drop_loss': p2_opt.metrics['distance_drop_loss'],
                'response_drop_loss': p2_opt.metrics['response_drop_loss'],
                'price_drop_loss': p2_opt.metrics['price_drop_loss'],
                'max_utilization': p2_opt.metrics['max_utilization'],
                'annual_profit_yuan_problem2': p2_opt.metrics['annual_profit_yuan'],
                'annual_subsidy_yuan_problem3': float(p3_opt.station_finance['annual_subsidy'].sum()),
                'avg_profit_rate_problem3': float(p3_opt.station_finance['profit_rate'].mean()),
                'station_stability_rate': stability,
                'scale_adjustment_rate': scale_adjust,
                'baseline_locations_under_scenario': ','.join(base_plan.keys()) if base_plan else '',
                'baseline_score_under_scenario': baseline_score_s,
                'optimal_score_under_scenario': optimal_score_s,
                'regret_vs_reoptimized_score': regret,
                'robustness_index': robustness_index,
                'candidate_plan_count': len(p2_summary),
                'baseline_annual_subsidy_under_scenario': (
                    float(baseline_p3_s.station_finance['annual_subsidy'].sum()) if baseline_p3_s is not None else None
                ),
            })
            plans[name] = {
                'p1': p1_s,
                'p2': p2_opt,
                'p2_summary': p2_summary,
                'p3': p3_opt,
                'baseline_eval': baseline_eval_s,
                'baseline_p3': baseline_p3_s,
            }
            print(f'[Problem4] scenario finished: {name}', flush=True)

        return SensitivityResult(pd.DataFrame(rows), plans)
