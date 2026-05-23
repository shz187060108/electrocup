from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from src.models.problem1_markov_demand import STATE_CN
from src.solvers.fixed_point import EndogenousSatisfactionSolver
from src.utils.metrics import price_satisfaction


@dataclass
class PricingResult:
    station_prices: pd.DataFrame
    station_finance: pd.DataFrame
    community_satisfaction: pd.DataFrame
    demand_release: pd.DataFrame
    assignment_table: pd.DataFrame
    station_table: pd.DataFrame


class PricingSubsidyOptimizer:
    """问题三：服务定价与政府补贴优化。

    完整版求解逻辑：
    1. 固定问题二得到的最优站点位置和规模；
    2. 对每个服务站进行五类收费服务的临界价格全组合枚举；
    3. 在保本和利润率不超过8%的约束下，选择价格满意度最高的价格向量；
    4. 重新计算价格变化后的消费约束需求、服务站选择、响应满意度和补贴；
    5. 输出潜在需求释放指数，用于分析不同类型老人服务可及性变化。
    """

    def __init__(self, raw, cfg: dict, p1_outputs, best_plan_eval):
        self.raw = raw
        self.cfg = cfg
        self.p1 = p1_outputs
        self.best = best_plan_eval
        self.p3_cfg = cfg['problem3']
        self.days_per_month = float(cfg['problem2']['days_per_month'])
        self.days_per_year = float(cfg['problem2']['days_per_year'])
        self.service_cost = raw.service_cost.set_index('service')
        self.services = list(raw.service_cost['service'])
        self.emergency = self.p3_cfg['emergency_service_name']
        self.non_emergency_services = [s for s in self.services if s != self.emergency]
        self._freq = raw.demand_freq.set_index('service')
        self._pop5 = p1_outputs.year5_population.set_index('community')
        self._base_price = raw.service_cost.set_index('service')['base_price'].to_dict()
        self._direct_cost = raw.service_cost.set_index('service')['direct_cost'].to_dict()
        self._station_response = best_plan_eval.fixed_point.station_table.set_index('station')['response_score'].to_dict()
        self._community_state_records = []
        for community, c in self._pop5.iterrows():
            for state_col, state_name in STATE_CN.items():
                self._community_state_records.append({
                    'community': community,
                    'state_name': state_name,
                    'n': float(c[state_col]),
                    'income': float(c['monthly_income']),
                    'cap_ratio': float(raw.consumption_caps[state_name]),
                    'freq': {s: float(self._freq.loc[s, state_name]) for s in self.services},
                })
        self.fp_solver = EndogenousSatisfactionSolver(
            raw.distance,
            cfg['problem2']['service_radius_m'],
            cfg['problem2']['fixed_point']['max_iter'],
            cfg['problem2']['fixed_point']['tolerance'],
        )

    def run(self) -> PricingResult:
        assignment = self.best.fixed_point.assignment_table.copy()
        station_price_rows = []
        station_finance_rows = []
        for station in self.best.plan.keys():
            assigned = assignment.loc[assignment['station'] == station, 'community'].tolist()
            best_price, finance = self._optimize_station_prices(station, assigned)
            for service, price in best_price.items():
                station_price_rows.append({
                    'station': station,
                    'service': service,
                    'base_price': float(self.service_cost.loc[service, 'base_price']),
                    'optimized_price': float(price),
                    'price_satisfaction': price_satisfaction(float(price), float(self.service_cost.loc[service, 'base_price'])),
                })
            station_finance_rows.append({'station': station, **finance})
        prices_df = pd.DataFrame(station_price_rows)
        finance_df = pd.DataFrame(station_finance_rows)

        community_price_map = self._community_prices_from_assignment(assignment, prices_df)
        budget_after_price, release_df = self._recompute_price_sensitive_demand(community_price_map)
        daily_by_community = (budget_after_price.groupby('community')['monthly_demand'].sum() / self.days_per_month).to_dict()
        elderly = self.p1.year5_population.set_index('community')['elderly_total'].to_dict()

        price_scores = self._community_station_price_scores(prices_df, budget_after_price)
        fp = self.fp_solver.solve(self.best.plan, daily_by_community, elderly, price_scores=price_scores)
        sat_df = fp.assignment_table[['community', 'station', 'distance_score', 'response_score', 'price_score', 'satisfaction']].copy()
        sat_df['elderly_total'] = sat_df['community'].map(elderly)
        return PricingResult(prices_df, finance_df, sat_df, release_df, fp.assignment_table, fp.station_table)

    def _candidate_prices(self, service: str) -> list[float]:
        base = float(self.service_cost.loc[service, 'base_price'])
        if service == self.emergency:
            return [float(self.p3_cfg['emergency_price'])]
        multipliers = list(self.p3_cfg['candidate_price_multipliers'])
        vals = [base * float(m) for m in multipliers]
        # 价格满意度的临界点。
        vals.extend([base, base * 1.10, base * 1.20])
        # 避免浮点重复，并保证价格非负。
        return sorted(set(round(v, 4) for v in vals if v >= 0))

    def _optimize_station_prices(self, station: str, assigned_communities: list[str]) -> Tuple[Dict[str, float], dict]:
        """临界价格全组合枚举。

        对助餐、日间照料、上门护理、康复理疗、助浴五类收费服务做笛卡尔积枚举；
        紧急救助按题设固定为公益免费，不参与定价，也不计补贴。
        """
        emergency_price = float(self.p3_cfg['emergency_price'])
        candidate_map = {s: self._candidate_prices(s) for s in self.non_emergency_services}
        services = list(self.non_emergency_services)
        candidate_lists = [candidate_map[s] for s in services]
        max_combinations = self.p3_cfg.get('pricing_search', {}).get('max_combinations_per_station')
        max_combinations = None if max_combinations in (None, 'null') else int(max_combinations)

        best_obj, best_prices, best_finance = None, None, None
        evaluated = 0
        for combo in product(*candidate_lists):
            evaluated += 1
            if max_combinations is not None and evaluated > max_combinations:
                break
            prices = {s: float(p) for s, p in zip(services, combo)}
            prices[self.emergency] = emergency_price
            finance = self._station_finance_under_prices(station, assigned_communities, prices)
            feasible = (
                finance['profit_rate'] >= float(self.p3_cfg['profit_rate_lower']) - 1e-9
                and finance['profit_rate'] <= float(self.p3_cfg['profit_rate_upper']) + 1e-9
            )
            lower = float(self.p3_cfg['profit_rate_lower'])
            upper = float(self.p3_cfg['profit_rate_upper'])
            if finance['profit_rate'] < lower:
                violation = lower - finance['profit_rate']
            elif finance['profit_rate'] > upper:
                violation = finance['profit_rate'] - upper
            else:
                violation = 0.0
            # 目标：先可行，再最大化需求加权价格满意度，再释放更多潜在需求，再选择更低价格。
            avg_price = float(np.mean([prices[s] for s in self.non_emergency_services])) if self.non_emergency_services else 0.0
            obj = (
                1 if feasible else 0,
                -violation,
                finance['weighted_price_satisfaction'],
                finance['released_monthly_demand'],
                -avg_price,
                -abs(finance['profit_rate']),
            )
            if best_obj is None or obj > best_obj:
                best_obj, best_prices, best_finance = obj, prices, finance

        if best_prices is None:
            best_prices = {s: float(self.service_cost.loc[s, 'base_price']) for s in self.non_emergency_services}
            best_prices[self.emergency] = emergency_price
            best_finance = self._station_finance_under_prices(station, assigned_communities, best_prices)
        best_finance = dict(best_finance)
        best_finance['price_combinations_evaluated'] = evaluated
        return best_prices, best_finance

    def _station_finance_under_prices(self, station: str, communities: list[str], prices: Dict[str, float]) -> dict:
        community_set = set(communities)
        demand_by_service = {s: 0.0 for s in self.services}
        base_budget_by_service = {s: 0.0 for s in self.services}
        base_budget = self.p1.budgeted_monthly_demand
        if communities:
            base_sub = base_budget[base_budget['community'].isin(communities)]
            base_budget_by_service.update(base_sub.groupby('service')['monthly_demand'].sum().to_dict())

        for rec in self._community_state_records:
            if rec['community'] not in community_set:
                continue
            per_cost = sum(rec['freq'][s] * float(prices[s]) for s in self.services)
            cap = rec['income'] * rec['cap_ratio']
            scale = min(1.0, cap / per_cost) if per_cost > 0 else 1.0
            for s in self.services:
                demand_by_service[s] += rec['n'] * rec['freq'][s] * scale

        effective_factor = float(self._station_response.get(station, 1.0))
        revenue = 0.0
        direct = 0.0
        non_emergency_eff_monthly = 0.0
        weighted_price_sat_num = 0.0
        weighted_price_sat_den = 0.0
        for s, q_month in demand_by_service.items():
            q_eff = q_month * effective_factor
            revenue += q_eff * float(prices[s]) * 12
            direct += q_eff * float(self._direct_cost[s]) * 12
            if s != self.emergency:
                non_emergency_eff_monthly += q_eff
                weighted_price_sat_num += q_eff * price_satisfaction(float(prices[s]), float(self._base_price[s]))
                weighted_price_sat_den += q_eff
        weighted_price_satisfaction = weighted_price_sat_num / weighted_price_sat_den if weighted_price_sat_den > 0 else 1.0
        released_monthly_demand = sum(demand_by_service[s] - base_budget_by_service.get(s, 0.0) for s in self.services)

        non_emergency_eff_daily = non_emergency_eff_monthly / self.days_per_month
        plan_info = self.best.plan[station]
        subsidy_cap_daily = {'小型': 1000, '中型': 1800, '大型': 2600}[plan_info['scale']]
        subsidy = min(float(self.p3_cfg['subsidy_per_effective_person_time']) * non_emergency_eff_daily, subsidy_cap_daily) * self.days_per_year
        fixed_mgmt = float(plan_info['fixed_daily_cost']) * self.days_per_year
        depreciation = float(plan_info['build_cost_wan']) * 10000 / 20
        annual_operating_cost = fixed_mgmt + depreciation
        service_profit = revenue - direct
        profit_after_subsidy = service_profit + subsidy - annual_operating_cost
        profit_rate = profit_after_subsidy / annual_operating_cost if annual_operating_cost > 0 else 0.0
        return {
            'annual_revenue': float(revenue),
            'annual_direct_cost': float(direct),
            'service_profit': float(service_profit),
            'annual_subsidy': float(subsidy),
            'annual_operating_cost': float(annual_operating_cost),
            'annual_profit_after_subsidy': float(profit_after_subsidy),
            'profit_rate': float(profit_rate),
            'pricing_feasible': int(0 <= profit_rate <= float(self.p3_cfg['profit_rate_upper'])),
            'weighted_price_satisfaction': float(weighted_price_satisfaction),
            'released_monthly_demand': float(released_monthly_demand),
        }

    def _community_prices_from_assignment(self, assignment: pd.DataFrame, prices_df: pd.DataFrame) -> Dict[tuple, float]:
        price_map = {(r.station, r.service): r.optimized_price for r in prices_df.itertuples()}
        result = {}
        for r in assignment.itertuples():
            if pd.isna(r.station):
                continue
            for s in self.services:
                result[(r.community, s)] = float(price_map[(r.station, s)])
        return result

    def _recompute_price_sensitive_demand(self, community_price_map: Dict[tuple, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        freq = self.raw.demand_freq.set_index('service')
        pop5 = self.p1.year5_population.set_index('community')
        base_budget = self.p1.budgeted_monthly_demand.copy()
        base_map = base_budget.groupby(['community', 'elder_type', 'service'])['monthly_demand'].sum().to_dict()
        theory_map = self.p1.theoretical_monthly_demand.groupby(['community', 'elder_type', 'service'])['monthly_demand'].sum().to_dict()
        rows, release_rows = [], []
        for community, c in pop5.iterrows():
            income = float(c['monthly_income'])
            for state_col, state_name in STATE_CN.items():
                n = float(c[state_col])
                per_cost = sum(
                    float(freq.loc[s, state_name])
                    * float(community_price_map.get((community, s), self.service_cost.loc[s, 'base_price']))
                    for s in self.services
                )
                cap = income * self.raw.consumption_caps[state_name]
                scale = min(1.0, cap / per_cost) if per_cost > 0 else 1.0
                for s in self.services:
                    q_new = n * float(freq.loc[s, state_name]) * scale
                    q_base = float(base_map.get((community, state_name, s), 0.0))
                    q_theory = float(theory_map.get((community, state_name, s), 0.0))
                    rows.append({'community': community, 'elder_type': state_name, 'service': s, 'monthly_demand': q_new})
                    denom = q_theory - q_base
                    release = (q_new - q_base) / denom if denom > 1e-9 else 0.0
                    release_rows.append({
                        'community': community,
                        'elder_type': state_name,
                        'service': s,
                        'latent_demand_release_index': max(0.0, min(1.0, release)),
                    })
        release_df = pd.DataFrame(release_rows).groupby(['community', 'elder_type'], as_index=False)['latent_demand_release_index'].mean()
        return pd.DataFrame(rows), release_df

    def _community_station_price_scores(self, prices_df: pd.DataFrame, demand: pd.DataFrame) -> Dict[tuple, float]:
        score_map = {(r.station, r.service): r.price_satisfaction for r in prices_df.itertuples()}
        service_mix = demand.groupby(['community', 'service'], as_index=False)['monthly_demand'].sum()
        total = service_mix.groupby('community')['monthly_demand'].sum().to_dict()
        result = {}
        for community in service_mix['community'].unique():
            sub = service_mix[service_mix['community'] == community]
            for station in self.best.plan.keys():
                s_val = 0.0
                for r in sub.itertuples():
                    w = r.monthly_demand / total[community] if total[community] > 0 else 0.0
                    s_val += w * float(score_map.get((station, r.service), 1.0))
                result[(community, station)] = float(s_val)
        return result
