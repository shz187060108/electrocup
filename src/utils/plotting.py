from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_population_trend(population_by_year: pd.DataFrame, out_path: str | Path) -> None:
    total = population_by_year.groupby('year', as_index=False)['elderly_total'].sum()
    plt.figure(figsize=(7, 4))
    plt.plot(total['year'], total['elderly_total'], marker='o')
    plt.xlabel('Year')
    plt.ylabel('Elderly population')
    plt.title('Five-year elderly population trend')
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_scenario_compare(scenario_summary: pd.DataFrame, out_path: str | Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.bar(scenario_summary['scenario'], scenario_summary['average_satisfaction'])
    plt.xticks(rotation=30, ha='right')
    plt.ylabel('Average satisfaction')
    plt.title('Scenario comparison')
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
