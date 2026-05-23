from __future__ import annotations

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path
import argparse
import pandas as pd

from src.config import load_config, ensure_dirs
from src.data.loaders import load_raw_data
from src.models.problem1_markov_demand import MarkovDemandModel
from src.models.problem2_cmclp_bilevel import LocationCapacityOptimizer
from src.models.problem3_pricing_subsidy import PricingSubsidyOptimizer
from src.models.problem4_sensitivity import SensitivityAnalyzer
from src.utils.io import save_dataframe, save_json
from src.utils.logging import get_logger
from src.utils.plotting import plot_population_trend, plot_scenario_compare


def main(config_path: str = 'configs/default.yaml'):
    cfg = load_config(config_path)
    ensure_dirs(cfg)
    od = Path(cfg['project']['output_dir'])
    fd = Path(cfg['project']['figure_dir'])
    logger = get_logger(log_file=str(Path(cfg['project']['log_dir']) / 'run_all.log'))

    raw = load_raw_data(cfg)

    logger.info(cfg['problem1']['name'])
    p1 = MarkovDemandModel(raw, cfg).run()
    save_dataframe(p1.population_by_year, od / 'problem1_population_by_year.csv')
    save_dataframe(p1.year5_population, od / 'problem1_year5_population.csv')
    save_dataframe(p1.theoretical_monthly_demand, od / 'problem1_theoretical_monthly_demand.csv')
    save_dataframe(p1.budgeted_monthly_demand, od / 'problem1_budgeted_monthly_demand.csv')
    save_dataframe(p1.scale_factors, od / 'problem1_consumption_scale_factors.csv')
    plot_population_trend(p1.population_by_year, fd / 'problem1_population_trend.png')

    logger.info(cfg['problem2']['name'])
    p2, p2_summary = LocationCapacityOptimizer(raw, cfg, p1).run()
    save_dataframe(p2_summary.head(500), od / 'problem2_top500_plan_summary.csv')
    save_dataframe(p2.fixed_point.assignment_table, od / 'problem2_best_assignment.csv')
    save_dataframe(p2.fixed_point.station_table, od / 'problem2_best_station_status.csv')
    save_dataframe(p2.metrics['station_profit_table'], od / 'problem2_best_station_profit.csv')
    save_json({k: v for k, v in p2.metrics.items() if k != 'station_profit_table'}, od / 'problem2_best_metrics.json')

    logger.info(cfg['problem3']['name'])
    p3 = PricingSubsidyOptimizer(raw, cfg, p1, p2).run()
    save_dataframe(p3.station_prices, od / 'problem3_station_prices.csv')
    save_dataframe(p3.station_finance, od / 'problem3_station_finance.csv')
    save_dataframe(p3.community_satisfaction, od / 'problem3_community_satisfaction.csv')
    save_dataframe(p3.demand_release, od / 'problem3_latent_demand_release.csv')
    save_dataframe(p3.assignment_table, od / 'problem3_assignment_after_pricing.csv')
    save_dataframe(p3.station_table, od / 'problem3_station_status_after_pricing.csv')

    logger.info(cfg['problem4']['name'])
    p4 = SensitivityAnalyzer(raw, cfg, p2).run()
    save_dataframe(p4.scenario_summary, od / 'problem4_scenario_summary.csv')
    plot_scenario_compare(p4.scenario_summary, fd / 'problem4_scenario_compare.png')

    # One Excel workbook for quick checking.
    with pd.ExcelWriter(od / 'all_results_summary.xlsx') as writer:
        p1.year5_population.to_excel(writer, sheet_name='问题一_第5年人口', index=False)
        p1.scale_factors.to_excel(writer, sheet_name='问题一_消费约束系数', index=False)
        p2_summary.head(100).to_excel(writer, sheet_name='问题二_方案Top100', index=False)
        p2.fixed_point.assignment_table.to_excel(writer, sheet_name='问题二_最优分配', index=False)
        p2.fixed_point.station_table.to_excel(writer, sheet_name='问题二_站点状态', index=False)
        p3.station_prices.to_excel(writer, sheet_name='问题三_最优定价', index=False)
        p3.station_finance.to_excel(writer, sheet_name='问题三_利润补贴', index=False)
        p3.demand_release.to_excel(writer, sheet_name='问题三_需求释放', index=False)
        p4.scenario_summary.to_excel(writer, sheet_name='问题四_情景比较', index=False)
    logger.info('All tasks finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()
    main(args.config)
