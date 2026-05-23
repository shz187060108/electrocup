from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from src.solvers.fixed_point import EndogenousSatisfactionSolver, FixedPointResult
from src.utils.metrics import weighted_mean


@dataclass
class PlanEvaluation:
    plan_id: int
    plan: Dict[str, dict]
    fixed_point: FixedPointResult
    metrics: dict


class LocationCapacityOptimizer:
    """问题二：服务站选址与规模优化。

    完整版求解逻辑：
    1. 枚举所有预算可行的站点位置与规模组合；
    2. 对每个组合运行下层“满意度最大选择 + 拥挤效应”固定点迭代；
    3. 计算覆盖率、平均满意度、最低满意度、三类满意度掉档损失、利用率和利润；
    4. 按多目标评分选择最优方案。

    该实现不再使用快速预筛选，适合小规模离散选址问题的可复现精确枚举。
    """

    def __init__(self, raw, cfg: dict, p1_outputs):
        self.raw = raw
        self.cfg = cfg
        self.p1 = p1_outputs
        self.p2_cfg = cfg['problem2']
        self.radius = float(self.p2_cfg['service_radius_m'])
        self.budget_wan = float(self.p2_cfg['budget_wan'])
        self.days_per_month = float(self.p2_cfg['days_per_month'])
        self.days_per_year = float(self.p2_cfg['days_per_year'])
        fp_cfg = self.p2_cfg['fixed_point']
        self.fp_solver = EndogenousSatisfactionSolver(
            raw.distance,
            self.radius,
            int(fp_cfg['max_iter']),
            float(fp_cfg['tolerance']),
        )
        self.communities = list(raw.communities['community'])
        self.scale_records = raw.station_cost.to_dict('records')
        self.scale_map = {r['scale']: r for r in self.scale_records}
        self.distance_lookup = {(str(i), str(j)): float(raw.distance.loc[i, j]) for i in raw.distance.index for j in raw.distance.columns}

        self.community_monthly_demand_by_service = (
            p1_outputs.budgeted_monthly_demand.groupby(['community', 'service'], as_index=False)['monthly_demand'].sum()
        )
        self.community_daily_demand = (
            self.community_monthly_demand_by_service.groupby('community')['monthly_demand'].sum() / self.days_per_month
        ).to_dict()
        self.community_elderly = p1_outputs.year5_population.set_index('community')['elderly_total'].to_dict()

    def enumerate_plans(self):
        """枚举所有满足建设预算的站点位置与规模组合。"""
        plan_id = 0
        n = len(self.communities)
        max_station_count = self.p2_cfg.get('enumeration', {}).get('max_station_count')
        for k in range(1, n + 1):
            if max_station_count is not None and k > int(max_station_count):
                break
            for locs in combinations(self.communities, k):
                # 至少覆盖一个小区，否则没有意义。
                if not any(
                    self.distance_lookup[(str(i), str(j))] <= self.radius
                    for i in self.communities
                    for j in locs
                ):
                    continue
                for scales in product(self.scale_records, repeat=k):
                    build_cost = sum(float(s['build_cost_wan']) for s in scales)
                    if build_cost > self.budget_wan:
                        continue
                    plan_id += 1
                    plan = {}
                    for j, s in zip(locs, scales):
                        plan[j] = {
                            'scale': s['scale'],
                            'capacity': float(s['daily_capacity']),
                            'build_cost_wan': float(s['build_cost_wan']),
                            'fixed_daily_cost': float(s['fixed_daily_cost']),
                        }
                    yield plan_id, plan

    def evaluate_plan(self, plan_id: int, plan: Dict[str, dict]) -> PlanEvaluation:
        fp = self.fp_solver.solve(plan, self.community_daily_demand, self.community_elderly)
        metrics = self._compute_metrics(plan, fp)
        return PlanEvaluation(plan_id, plan, fp, metrics)

    def _compute_metrics(self, plan: Dict[str, dict], fp: FixedPointResult) -> dict:
        assign = fp.assignment_table.copy()
        pop = assign['community'].map(self.community_elderly).astype(float)
        total_pop = float(sum(self.community_elderly.values()))
        covered_pop = float((pop * assign['covered']).sum())
        coverage_rate = covered_pop / total_pop if total_pop > 0 else 0.0

        avg_sat = weighted_mean(assign['satisfaction'], pop)
        min_sat = float(assign.loc[assign['covered'] == 1, 'satisfaction'].min()) if (assign['covered'] == 1).any() else 0.0

        loss_targets = self.p2_cfg.get('loss_targets', {})
        target_total = float(loss_targets.get('comprehensive', self.p2_cfg.get('satisfaction_target', 0.85)))
        target_distance = float(loss_targets.get('distance', 0.75))
        target_response = float(loss_targets.get('response', 0.85))
        target_price = float(loss_targets.get('price', 1.00))

        comprehensive_drop_loss = float((pop * np.maximum(0, target_total - assign['satisfaction'])).sum())
        distance_drop_loss = float((pop * np.maximum(0, target_distance - assign['distance_score'])).sum())
        response_drop_loss = float((pop * np.maximum(0, target_response - assign['response_score'])).sum())
        price_drop_loss = float((pop * np.maximum(0, target_price - assign['price_score'])).sum())

        component_weights = self.p2_cfg.get('component_loss_weights', {'distance': 0.2, 'response': 0.3, 'price': 0.5})
        threshold_drop_loss = (
            float(component_weights.get('distance', 0.2)) * distance_drop_loss
            + float(component_weights.get('response', 0.3)) * response_drop_loss
            + float(component_weights.get('price', 0.5)) * price_drop_loss
        )

        util_values = np.array(list(fp.station_utilization.values()), dtype=float)
        max_util = float(util_values.max()) if len(util_values) else 0.0
        overload = float(np.maximum(0, util_values - 1.0).sum()) if len(util_values) else 0.0
        high_util_penalty = float(np.maximum(0, util_values - 0.85).sum()) if len(util_values) else 0.0
        profit, station_profit = self._estimate_annual_profit(plan, fp)

        weights = self.p2_cfg['weights']
        score = (
            float(weights['coverage']) * coverage_rate
            + float(weights['average_satisfaction']) * avg_sat
            + float(weights['min_satisfaction']) * min_sat
            + float(weights['drop_loss']) * comprehensive_drop_loss
            + float(weights.get('threshold_drop_loss', weights['drop_loss'])) * threshold_drop_loss
            + float(weights['utilization_penalty']) * high_util_penalty
            + float(weights['profit']) * profit
            - 1e6 * overload
        )

        return {
            'coverage_rate': coverage_rate,
            'covered_elderly': covered_pop,
            'average_satisfaction': avg_sat,
            'min_satisfaction': min_sat,
            'drop_loss': comprehensive_drop_loss,
            'threshold_drop_loss': threshold_drop_loss,
            'distance_drop_loss': distance_drop_loss,
            'response_drop_loss': response_drop_loss,
            'price_drop_loss': price_drop_loss,
            'max_utilization': max_util,
            'overload': overload,
            'high_util_penalty': high_util_penalty,
            'annual_profit_yuan': profit,
            'score': score,
            'station_count': len(plan),
            'total_capacity': float(sum(p['capacity'] for p in plan.values())),
            'total_build_cost_wan': float(sum(p['build_cost_wan'] for p in plan.values())),
            'station_profit_table': station_profit,
        }

    def _estimate_annual_profit(self, plan: Dict[str, dict], fp: FixedPointResult) -> Tuple[float, pd.DataFrame]:
        demand = self.community_monthly_demand_by_service.copy()
        assignment = fp.assignment_table.set_index('community')
        demand['station'] = demand['community'].map(assignment['station'])
        demand['satisfaction'] = demand['community'].map(assignment['satisfaction'])
        demand = demand.dropna(subset=['station'])
        demand['effective_monthly_demand'] = demand['monthly_demand'] * demand['satisfaction']
        cost = self.raw.service_cost.set_index('service')
        demand['base_price'] = demand['service'].map(cost['base_price'])
        demand['direct_cost'] = demand['service'].map(cost['direct_cost'])
        demand['annual_revenue'] = demand['effective_monthly_demand'] * demand['base_price'] * 12
        demand['annual_direct_cost'] = demand['effective_monthly_demand'] * demand['direct_cost'] * 12
        station_profit = demand.groupby('station', as_index=False)[['annual_revenue', 'annual_direct_cost']].sum()
        station_profit['fixed_management_cost'] = station_profit['station'].map({j: p['fixed_daily_cost'] * self.days_per_year for j, p in plan.items()})
        station_profit['annual_depreciation'] = station_profit['station'].map({j: p['build_cost_wan'] * 10000 / 20 for j, p in plan.items()})
        station_profit['annual_profit_yuan'] = (
            station_profit['annual_revenue']
            - station_profit['annual_direct_cost']
            - station_profit['fixed_management_cost']
            - station_profit['annual_depreciation']
        )
        for j, p in plan.items():
            if j not in set(station_profit['station']):
                fixed = p['fixed_daily_cost'] * self.days_per_year
                dep = p['build_cost_wan'] * 10000 / 20
                station_profit = pd.concat([station_profit, pd.DataFrame([{
                    'station': j,
                    'annual_revenue': 0.0,
                    'annual_direct_cost': 0.0,
                    'fixed_management_cost': fixed,
                    'annual_depreciation': dep,
                    'annual_profit_yuan': -(fixed + dep),
                }])], ignore_index=True)
        return float(station_profit['annual_profit_yuan'].sum()), station_profit

    def run(self) -> Tuple[PlanEvaluation, pd.DataFrame]:
        show_progress = bool(self.p2_cfg.get('enumeration', {}).get('show_progress', False))
        progress_every = int(self.p2_cfg.get('enumeration', {}).get('progress_every', 1000))
        best: PlanEvaluation | None = None
        rows = []
        evaluated = 0
        feasible = 0

        for plan_id, plan in self.enumerate_plans():
            feasible += 1
            ev = self.evaluate_plan(plan_id, plan)
            evaluated += 1
            m = ev.metrics
            row = {k: v for k, v in m.items() if k != 'station_profit_table'}
            row.update({
                'plan_id': plan_id,
                'locations': ','.join(plan.keys()),
                'scales': ','.join([plan[j]['scale'] for j in plan.keys()]),
            })
            rows.append(row)
            if best is None or ev.metrics['score'] > best.metrics['score']:
                best = ev
            if show_progress and evaluated % progress_every == 0:
                print(f'[Problem2] full fixed-point evaluated {evaluated} feasible plans...', flush=True)

        if best is None:
            raise RuntimeError('No feasible plan found under the current budget.')
        if show_progress:
            print(f'[Problem2] full enumeration finished: {evaluated} feasible plans evaluated.', flush=True)
        summary = pd.DataFrame(rows).sort_values('score', ascending=False).reset_index(drop=True)
        return best, summary
