from __future__ import annotations

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path
import argparse

from src.config import load_config, ensure_dirs
from src.data.loaders import load_raw_data
from src.models.problem1_markov_demand import MarkovDemandModel
from src.models.problem2_cmclp_bilevel import LocationCapacityOptimizer
from src.models.problem3_pricing_subsidy import PricingSubsidyOptimizer
from src.utils.io import save_dataframe
from src.utils.logging import get_logger


def main(config_path: str = 'configs/default.yaml'):
    cfg = load_config(config_path)
    ensure_dirs(cfg)
    logger = get_logger(log_file=str(Path(cfg['project']['log_dir']) / 'run_problem3.log'))
    logger.info(cfg['problem3']['name'])
    raw = load_raw_data(cfg)
    p1 = MarkovDemandModel(raw, cfg).run()
    p2, _ = LocationCapacityOptimizer(raw, cfg, p1).run()
    p3 = PricingSubsidyOptimizer(raw, cfg, p1, p2).run()
    od = Path(cfg['project']['output_dir'])
    save_dataframe(p3.station_prices, od / 'problem3_station_prices.csv')
    save_dataframe(p3.station_finance, od / 'problem3_station_finance.csv')
    save_dataframe(p3.community_satisfaction, od / 'problem3_community_satisfaction.csv')
    save_dataframe(p3.demand_release, od / 'problem3_latent_demand_release.csv')
    save_dataframe(p3.assignment_table, od / 'problem3_assignment_after_pricing.csv')
    save_dataframe(p3.station_table, od / 'problem3_station_status_after_pricing.csv')
    logger.info('Saved Problem 3 outputs.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()
    main(args.config)
