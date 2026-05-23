from src.config import load_config
from src.data.loaders import load_raw_data
from src.models.problem1_markov_demand import MarkovDemandModel


def test_problem1_smoke():
    cfg = load_config('configs/default.yaml')
    raw = load_raw_data(cfg)
    out = MarkovDemandModel(raw, cfg).run()
    assert len(out.year5_population) == 10
    assert out.year5_population['elderly_total'].sum() > raw.communities['elderly_total'].sum()
    assert out.budgeted_monthly_demand['monthly_demand'].sum() <= out.theoretical_monthly_demand['monthly_demand'].sum() + 1e-6
