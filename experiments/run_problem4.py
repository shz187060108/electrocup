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
from src.models.problem4_sensitivity import SensitivityAnalyzer
from src.utils.io import save_dataframe
from src.utils.logging import get_logger
from src.utils.plotting import plot_scenario_compare


def main(config_path: str = 'configs/default.yaml'):
    cfg = load_config(config_path)
    ensure_dirs(cfg)
    logger = get_logger(log_file=str(Path(cfg['project']['log_dir']) / 'run_problem4.log'))
    logger.info(cfg['problem4']['name'])
    raw = load_raw_data(cfg)
    p1 = MarkovDemandModel(raw, cfg).run()
    p2, _ = LocationCapacityOptimizer(raw, cfg, p1).run()
    res = SensitivityAnalyzer(raw, cfg, p2).run()
    od = Path(cfg['project']['output_dir'])
    save_dataframe(res.scenario_summary, od / 'problem4_scenario_summary.csv')
    plot_scenario_compare(res.scenario_summary, Path(cfg['project']['figure_dir']) / 'problem4_scenario_compare.png')
    logger.info('Saved Problem 4 outputs.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()
    main(args.config)
