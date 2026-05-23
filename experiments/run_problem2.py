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
from src.utils.io import save_dataframe, save_json
from src.utils.logging import get_logger


def main(config_path: str = 'configs/default.yaml'):
    cfg = load_config(config_path)
    ensure_dirs(cfg)
    logger = get_logger(log_file=str(Path(cfg['project']['log_dir']) / 'run_problem2.log'))
    logger.info(cfg['problem2']['name'])
    raw = load_raw_data(cfg)
    p1 = MarkovDemandModel(raw, cfg).run()
    best, summary = LocationCapacityOptimizer(raw, cfg, p1).run()
    od = Path(cfg['project']['output_dir'])
    save_dataframe(summary.head(200), od / 'problem2_top200_plan_summary.csv')
    save_dataframe(best.fixed_point.assignment_table, od / 'problem2_best_assignment.csv')
    save_dataframe(best.fixed_point.station_table, od / 'problem2_best_station_status.csv')
    save_dataframe(best.metrics['station_profit_table'], od / 'problem2_best_station_profit.csv')
    metrics_clean = {k: v for k, v in best.metrics.items() if k != 'station_profit_table'}
    save_json(metrics_clean, od / 'problem2_best_metrics.json')
    logger.info(f"Best plan: {','.join(best.plan.keys())}; metrics={metrics_clean}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()
    main(args.config)
