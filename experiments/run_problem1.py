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
from src.utils.io import save_dataframe
from src.utils.logging import get_logger
from src.utils.plotting import plot_population_trend


def main(config_path: str = 'configs/default.yaml'):
    cfg = load_config(config_path)
    ensure_dirs(cfg)
    logger = get_logger(log_file=str(Path(cfg['project']['log_dir']) / 'run_problem1.log'))
    logger.info(cfg['problem1']['name'])
    raw = load_raw_data(cfg)
    out = MarkovDemandModel(raw, cfg).run()
    od = Path(cfg['project']['output_dir'])
    save_dataframe(out.population_by_year, od / 'problem1_population_by_year.csv')
    save_dataframe(out.year5_population, od / 'problem1_year5_population.csv')
    save_dataframe(out.theoretical_monthly_demand, od / 'problem1_theoretical_monthly_demand.csv')
    save_dataframe(out.budgeted_monthly_demand, od / 'problem1_budgeted_monthly_demand.csv')
    save_dataframe(out.scale_factors, od / 'problem1_consumption_scale_factors.csv')
    plot_population_trend(out.population_by_year, Path(cfg['project']['figure_dir']) / 'problem1_population_trend.png')
    logger.info('Saved Problem 1 outputs.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()
    main(args.config)
