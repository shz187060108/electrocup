from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import re
import numpy as np
import pandas as pd


@dataclass
class RawData:
    communities: pd.DataFrame
    transitions: Dict[str, float]
    demand_freq: pd.DataFrame
    service_cost: pd.DataFrame
    consumption_caps: Dict[str, float]
    station_cost: pd.DataFrame
    distance: pd.DataFrame


def _read_excel(path: Path, sheet_name: str, header: int = 1) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet_name, header=header)


def _parse_percent_text(x) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    m = re.search(r'(\d+(?:\.\d+)?)\s*%', s)
    if m:
        return float(m.group(1)) / 100.0
    m = re.search(r'(\d+(?:\.\d+)?)', s)
    return float(m.group(1)) if m else np.nan


def load_raw_data(cfg: dict) -> RawData:
    raw_dir = Path(cfg['data']['raw_dir'])

    a1 = raw_dir / cfg['data']['attachment1']
    communities = _read_excel(a1, '人口与老人结构', header=1)
    communities.columns = [str(c).strip() for c in communities.columns]
    communities = communities.rename(columns={
        '小区编号': 'community',
        '总人口': 'total_population',
        '60+老人数': 'elderly_total',
        '自理老人': 'self_care',
        '半失能老人': 'semi_disabled',
        '失能老人': 'disabled',
        '人均月收入(元)': 'monthly_income',
    })
    communities = communities.dropna(subset=['community']).reset_index(drop=True)

    trans_df = pd.read_excel(a1, sheet_name='转移概率', header=1).dropna()
    transitions = {str(r['转移类型']).strip(): float(r['年度转移概率参考区间']) for _, r in trans_df.iterrows()}

    a2 = raw_dir / cfg['data']['attachment2']
    demand_freq = _read_excel(a2, '每位老人月均服务需求次数', header=1)
    demand_freq.columns = ['service', '自理老人', '半失能老人', '失能老人']
    demand_freq['service'] = demand_freq['service'].astype(str).str.strip()

    service_cost = _read_excel(a2, '服务营收及支出', header=1)
    service_cost.columns = ['service', 'base_price', 'direct_cost']
    service_cost['service'] = service_cost['service'].astype(str).str.strip()
    service_cost['base_price'] = service_cost['base_price'].replace({'0（公益免费）': 0}).astype(float)
    service_cost['direct_cost'] = service_cost['direct_cost'].astype(float)

    caps_df = pd.read_excel(a2, sheet_name='月服务消费上限', header=0).iloc[:3]
    consumption_caps = {
        str(row['老人类型']).strip(): _parse_percent_text(row['月服务消费上限（占月收入比例）'])
        for _, row in caps_df.iterrows()
        if pd.notna(row['老人类型'])
    }

    a3 = raw_dir / cfg['data']['attachment3']
    station_cost = _read_excel(a3, '服务站建设与运营成本', header=1)
    station_cost = station_cost.iloc[:3].copy()
    station_cost.columns = ['scale', 'build_cost_wan', 'fixed_daily_cost', 'daily_capacity']
    station_cost['scale'] = station_cost['scale'].astype(str).str.strip()
    station_cost[['build_cost_wan', 'fixed_daily_cost', 'daily_capacity']] = station_cost[['build_cost_wan', 'fixed_daily_cost', 'daily_capacity']].astype(float)

    a4 = raw_dir / cfg['data']['attachment4']
    distance = _read_excel(a4, '小区间距离矩阵', header=1)
    distance = distance.rename(columns={'组别': 'community'})
    distance['community'] = distance['community'].astype(str).str.strip()
    distance = distance.set_index('community')
    distance.columns = [str(c).strip() for c in distance.columns]
    distance = distance.astype(float)

    return RawData(
        communities=communities,
        transitions=transitions,
        demand_freq=demand_freq,
        service_cost=service_cost,
        consumption_caps=consumption_caps,
        station_cost=station_cost,
        distance=distance,
    )
