from __future__ import annotations

import numpy as np


def distance_satisfaction(distance_m: float) -> float:
    if distance_m <= 300:
        return 1.00
    if distance_m <= 500:
        return 0.90
    if distance_m <= 650:
        return 0.75
    if distance_m <= 1000:
        return 0.60
    return 0.0


def response_satisfaction(utilization: float) -> float:
    if utilization <= 0.60:
        return 1.00
    if utilization <= 0.75:
        return 0.93
    if utilization <= 0.85:
        return 0.85
    if utilization <= 0.95:
        return 0.72
    if utilization <= 1.00:
        return 0.60
    # 超容量时保留最低响应满意度，并在目标函数中另行惩罚。
    return 0.60


def price_satisfaction(price: float, base_price: float) -> float:
    if base_price <= 0:
        return 1.00 if price <= 0 else 0.60
    ratio = price / base_price
    if ratio <= 1.00:
        return 1.00
    if ratio <= 1.10:
        return 0.90
    if ratio <= 1.20:
        return 0.75
    return 0.60


def weighted_mean(values, weights) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    s = weights.sum()
    return float((values * weights).sum() / s) if s > 0 else 0.0
