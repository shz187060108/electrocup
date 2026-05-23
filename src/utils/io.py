from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_dataframe(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    p = ensure_parent(path)
    suffix = p.suffix.lower()
    if suffix == '.csv':
        df.to_csv(p, index=index, encoding='utf-8-sig')
    elif suffix in ['.xlsx', '.xls']:
        df.to_excel(p, index=index)
    else:
        raise ValueError(f'Unsupported table format: {suffix}')


def save_json(obj, path: str | Path) -> None:
    p = ensure_parent(path)
    with p.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
