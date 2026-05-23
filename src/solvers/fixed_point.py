from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from src.utils.metrics import distance_satisfaction, response_satisfaction


@dataclass
class FixedPointResult:
    assignment: Dict[str, str]
    community_satisfaction: Dict[str, float]
    station_utilization: Dict[str, float]
    station_response_score: Dict[str, float]
    iterations: int
    converged: bool
    assignment_table: pd.DataFrame
    station_table: pd.DataFrame


class EndogenousSatisfactionSolver:
    """老人满意度最大选择规则 + 拥挤效应固定点迭代。

    该求解器对应双层规划模型中的下层选择问题。给定上层站点位置、规模与价格，
    小区老人选择综合满意度最高的可达服务站；选择结果改变服务站利用率，进而改变
    响应满意度，最终通过固定点迭代获得稳定分配。
    """

    def __init__(self, distance: pd.DataFrame, service_radius_m: float, max_iter: int = 60, tol: float = 1e-9):
        self.distance = distance
        self.distance_lookup = {(str(i), str(j)): float(distance.loc[i, j]) for i in distance.index for j in distance.columns}
        self.radius = float(service_radius_m)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def solve(
        self,
        plan: Dict[str, dict],
        community_daily_demand: Dict[str, float],
        community_elderly: Dict[str, float],
        price_scores: Dict[Tuple[str, str], float] | None = None,
    ) -> FixedPointResult:
        """固定点迭代求解。

        Args:
            plan: {station_community: {'scale': str, 'capacity': float, ...}}
            community_daily_demand: 消费约束或价格约束后的日需求。
            community_elderly: 各小区第5年老人数量，用于覆盖率和加权满意度。
            price_scores: {(community, station): score}，默认全部为1。
        """
        stations = list(plan.keys())
        communities = list(community_daily_demand.keys())
        response_scores = {j: 1.0 for j in stations}
        util = {j: 0.0 for j in stations}
        assignment: Dict[str, str | None] = {}
        sat: Dict[str, float] = {}
        component_scores: Dict[str, dict] = {}
        converged = False

        for it in range(1, self.max_iter + 1):
            new_assignment: Dict[str, str | None] = {}
            new_sat: Dict[str, float] = {}
            new_components: Dict[str, dict] = {}

            for i in communities:
                candidates = []
                for j in stations:
                    d = self.distance_lookup[(str(i), str(j))]
                    if d <= self.radius:
                        s_distance = distance_satisfaction(d)
                        s_response = response_scores[j]
                        s_price = 1.0 if price_scores is None else float(price_scores.get((i, j), 1.0))
                        s_total = 0.2 * s_distance + 0.3 * s_response + 0.5 * s_price
                        # tie-breaking: 综合满意度、距离满意度、近距离、低利用率、站点编号。
                        candidates.append((s_total, s_distance, -d, -util[j], str(j), j, s_response, s_price))
                if candidates:
                    best = sorted(candidates, reverse=True)[0]
                    s_total, s_distance, neg_d, neg_util, _, j, s_response, s_price = best
                    new_assignment[i] = j
                    new_sat[i] = float(s_total)
                    new_components[i] = {
                        'distance_score': float(s_distance),
                        'response_score': float(s_response),
                        'price_score': float(s_price),
                        'distance_m': float(-neg_d),
                    }
                else:
                    new_assignment[i] = None
                    new_sat[i] = 0.0
                    new_components[i] = {
                        'distance_score': 0.0,
                        'response_score': 0.0,
                        'price_score': 0.0,
                        'distance_m': np.nan,
                    }

            effective_load = {j: 0.0 for j in stations}
            raw_load = {j: 0.0 for j in stations}
            for i, j in new_assignment.items():
                if j is None:
                    continue
                raw_q = float(community_daily_demand[i])
                raw_load[j] += raw_q
                effective_load[j] += raw_q * new_sat[i]

            new_util = {j: effective_load[j] / float(plan[j]['capacity']) for j in stations}
            new_response = {j: response_satisfaction(new_util[j]) for j in stations}

            diff = max(abs(new_response[j] - response_scores[j]) for j in stations) if stations else 0.0
            same_assignment = (new_assignment == assignment) if assignment else False
            assignment = new_assignment
            sat = new_sat
            component_scores = new_components
            util = new_util
            response_scores = new_response
            if same_assignment and diff <= self.tol:
                converged = True
                break

        assign_rows = []
        for i in communities:
            j = assignment.get(i)
            c = component_scores.get(i, {})
            if j is None:
                assign_rows.append({
                    'community': i,
                    'station': None,
                    'distance_m': np.nan,
                    'daily_demand': float(community_daily_demand[i]),
                    'distance_score': 0.0,
                    'response_score': 0.0,
                    'price_score': 0.0,
                    'satisfaction': 0.0,
                    'effective_daily_demand': 0.0,
                    'covered': 0,
                })
            else:
                assign_rows.append({
                    'community': i,
                    'station': j,
                    'distance_m': c['distance_m'],
                    'daily_demand': float(community_daily_demand[i]),
                    'distance_score': c['distance_score'],
                    'response_score': response_scores[j],
                    'price_score': c['price_score'],
                    'satisfaction': float(sat[i]),
                    'effective_daily_demand': float(community_daily_demand[i]) * float(sat[i]),
                    'covered': 1,
                })
        assignment_table = pd.DataFrame(assign_rows)

        station_rows = []
        for j in stations:
            station_rows.append({
                'station': j,
                'scale': plan[j]['scale'],
                'capacity': float(plan[j]['capacity']),
                'build_cost_wan': float(plan[j]['build_cost_wan']),
                'fixed_daily_cost': float(plan[j]['fixed_daily_cost']),
                'utilization': float(util[j]),
                'response_score': float(response_scores[j]),
                'assigned_communities': ','.join([i for i, jj in assignment.items() if jj == j]),
            })
        station_table = pd.DataFrame(station_rows)
        return FixedPointResult(assignment, sat, util, response_scores, it, converged, assignment_table, station_table)
