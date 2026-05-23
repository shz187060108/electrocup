from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd

STATE_COLS = ['self_care', 'semi_disabled', 'disabled']
STATE_CN = {'self_care': '自理老人', 'semi_disabled': '半失能老人', 'disabled': '失能老人'}
CN_TO_STATE = {v: k for k, v in STATE_CN.items()}


@dataclass
class Problem1Outputs:
    population_by_year: pd.DataFrame
    year5_population: pd.DataFrame
    theoretical_monthly_demand: pd.DataFrame
    budgeted_monthly_demand: pd.DataFrame
    demand_summary: pd.DataFrame
    scale_factors: pd.DataFrame


class MarkovDemandModel:
    """问题一：未来五年老人数量与服务需求量预测。

    使用多状态 Markov 人口递推，并按消费约束等比例修正服务需求。
    """

    def __init__(self, raw, cfg: dict):
        self.raw = raw
        self.cfg = cfg
        self.years = int(cfg['problem1']['years'])
        self.death_rate = float(cfg['problem1']['death_rate'])
        self.new_rate = float(cfg['problem1']['new_elderly_rate'])
        self.p12 = float(raw.transitions.get('自理 → 半失能', 0.0))
        self.p23 = float(raw.transitions.get('半失能 → 失能', 0.0))
        self.services = list(raw.demand_freq['service'])

    def predict_population(self) -> pd.DataFrame:
        rows = []
        base = self.raw.communities.copy()
        current = base[['community', 'monthly_income'] + STATE_COLS].copy()
        for _, r in current.iterrows():
            rows.append(self._population_row(0, r))
        for year in range(1, self.years + 1):
            next_rows = []
            for _, r in current.iterrows():
                self_care = float(r['self_care'])
                semi = float(r['semi_disabled'])
                disabled = float(r['disabled'])
                current_total = self_care + semi + disabled

                # 1. 状态转移
                move12 = self_care * self.p12
                move23 = semi * self.p23
                self_care_t = self_care - move12
                semi_t = semi + move12 - move23
                disabled_t = disabled + move23

                # 2. 自然死亡扣减
                survive = 1.0 - self.death_rate
                self_care_t *= survive
                semi_t *= survive
                disabled_t *= survive

                # 3. 新增老人归入自理状态
                self_care_t += current_total * self.new_rate

                new_r = {
                    'community': r['community'],
                    'monthly_income': r['monthly_income'],
                    'self_care': self_care_t,
                    'semi_disabled': semi_t,
                    'disabled': disabled_t,
                }
                next_rows.append(new_r)
                rows.append(self._population_row(year, pd.Series(new_r)))
            current = pd.DataFrame(next_rows)
        return pd.DataFrame(rows)

    @staticmethod
    def _population_row(year: int, r: pd.Series) -> dict:
        total = float(r['self_care']) + float(r['semi_disabled']) + float(r['disabled'])
        return {
            'year': year,
            'community': r['community'],
            'self_care': float(r['self_care']),
            'semi_disabled': float(r['semi_disabled']),
            'disabled': float(r['disabled']),
            'elderly_total': total,
            'monthly_income': float(r['monthly_income']),
        }

    def compute_demand(self, pop_year5: pd.DataFrame, prices: Dict[str, float] | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return theoretical demand, budgeted demand and scaling factors.

        Demand is measured by monthly service person-times.
        """
        freq = self.raw.demand_freq.set_index('service')
        cost = self.raw.service_cost.set_index('service')
        if prices is None:
            prices = cost['base_price'].to_dict()

        theo_rows = []
        budget_rows = []
        scale_rows = []
        for _, c in pop_year5.iterrows():
            income = float(c['monthly_income'])
            for state_col, state_name in STATE_CN.items():
                n_elderly = float(c[state_col])
                # Theoretical cost per elder under given prices.
                per_elder_cost = 0.0
                for service in self.services:
                    per_elder_cost += float(freq.loc[service, state_name]) * float(prices.get(service, cost.loc[service, 'base_price']))
                cap_ratio = self.raw.consumption_caps[state_name]
                cap_amount = income * cap_ratio
                scale = min(1.0, cap_amount / per_elder_cost) if per_elder_cost > 0 else 1.0
                scale_rows.append({
                    'community': c['community'], 'elder_type': state_name,
                    'monthly_income': income, 'cap_ratio': cap_ratio,
                    'theoretical_cost_per_elder': per_elder_cost,
                    'cap_amount_per_elder': cap_amount,
                    'scale_factor': scale,
                })
                for service in self.services:
                    q = n_elderly * float(freq.loc[service, state_name])
                    theo_rows.append({
                        'community': c['community'], 'elder_type': state_name,
                        'service': service, 'monthly_demand': q,
                    })
                    budget_rows.append({
                        'community': c['community'], 'elder_type': state_name,
                        'service': service, 'monthly_demand': q * scale,
                    })
        return pd.DataFrame(theo_rows), pd.DataFrame(budget_rows), pd.DataFrame(scale_rows)

    @staticmethod
    def summarize_demand(demand: pd.DataFrame, label: str) -> pd.DataFrame:
        return (demand.groupby(['community', 'service'], as_index=False)['monthly_demand']
                .sum().assign(demand_type=label))

    def run(self) -> Problem1Outputs:
        pop = self.predict_population()
        pop5 = pop[pop['year'] == self.years].copy().reset_index(drop=True)
        theo, budget, scale = self.compute_demand(pop5)
        summary = pd.concat([
            self.summarize_demand(theo, 'theoretical'),
            self.summarize_demand(budget, 'budgeted'),
        ], ignore_index=True)
        return Problem1Outputs(
            population_by_year=pop,
            year5_population=pop5,
            theoretical_monthly_demand=theo,
            budgeted_monthly_demand=budget,
            demand_summary=summary,
            scale_factors=scale,
        )
