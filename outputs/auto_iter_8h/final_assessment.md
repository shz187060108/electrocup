# Auto Iteration Assessment

Run window: 2026-05-23 04:22 to 2026-05-23 12:25, more than 8 hours.

## What changed

The original `run_all.py` completed, but its problem 2 scalar score selected a one-station plan at community A. That plan minimizes overload but covers only 44.99% of elderly residents, so it is not a strong competition answer for a planning problem whose stated objective is high coverage and high satisfaction.

I added `experiments/auto_iterative_optimizer.py`. It keeps the original exact enumeration engine, but evaluates several service-first multi-objective policies and writes auditable outputs under `outputs/auto_iter_8h`.

## Recommended baseline plan

Policy: `coverage_capacity`

Stations:

- C, small
- D, large
- G, large

Problem 2 metrics:

- Coverage rate: 100%
- Average satisfaction: 0.862214
- Minimum satisfaction: 0.800000
- Build cost: 108 wan
- Total capacity: 7000 person-times/day
- Max utilization: 1.034489
- Overload sum: 0.049846

Assignment:

- C station: C
- D station: A, B, D, H, J
- G station: E, F, G, I

## Pricing after low-fine grid refinement

Price profile: `low_fine_grid`, 759375 combinations evaluated per station.

All three stations remain pricing-feasible. Price satisfaction remains 1.0.

Station C:

- Profit rate: 0.0835%
- Annual subsidy: 365000 yuan
- Annual profit after subsidy: 617.22 yuan

Station D:

- Profit rate: 2.9881%
- Annual subsidy: 949000 yuan
- Annual profit after subsidy: 48660.87 yuan

Station G:

- Profit rate: 1.4516%
- Annual subsidy: 949000 yuan
- Annual profit after subsidy: 23639.06 yuan

Total annual subsidy: 2263000 yuan.

## Why this is stronger than the original output

The original output chose one large station at A:

- Coverage rate: 44.99%
- Average satisfaction: 0.376597
- Max utilization: 1.016331

That result is mathematically optimal only for the old scalar score, not for the planning objective in the problem statement. The improved recommendation accepts a slightly higher overload, but covers every community and every elderly resident within the model's service-radius rule.

## Remaining model risk

No enumerated baseline plan has zero overload under the current solver. This means the present model is internally capacity-tight. The response score is clamped at 0.6 after overload, so heavily overloaded close-distance plans can still look artificially satisfying. Future iterations should add either capacity-constrained assignment or a sharper overcapacity response curve.

The current satisfaction rules are hard-coded in `src/utils/metrics.py`. Attachment 5 exists and should be parsed directly to make the model more defensible in a paper setting.

Problem 3 is exact only over the specified discrete price grid. The low-fine grid improved prices, but a branch-and-bound or monotone search would be more efficient and allow denser price resolution.

## Recommended sensitivity results

I also ran the improved `coverage_capacity` policy across all five problem 4 scenarios with the base price grid. Outputs are under `outputs/auto_iter_p4_recommended_full`.

Summary:

- Baseline: C,D,G with small,large,large; coverage 100%; average satisfaction 0.862214; overload 0.049846.
- Elderly growth 8%: B,C,D with medium,large,medium; coverage 100%; average satisfaction 0.858939; overload 0.149577.
- Transition-up: C,H,I with small,large,large; coverage 100%; average satisfaction 0.859627; overload 0.088463.
- Fixed cost +20%: C,D,G remains selected; coverage and satisfaction unchanged; annual profit drops.
- Budget 140: C,D,G remains selected under the service-first/capacity-first policy; larger budget does not improve the selected trade-off because the policy first minimizes overload among full-coverage plans.

## Capacity-Hard Addendum

After reviewing the external B-D-F-J answer, I added a stricter capacity-hard evaluation in `experiments/compare_capacity_plans.py`.

This newer evaluation separates:

- spatial coverage;
- capacity fulfillment;
- effective service rate;
- per-community fairness.

Under this stricter model, the external B-D-F-J plan is feasible, but it is not the best plan in this project's data and scoring rules.

With a minimum 80% fulfillment constraint for every community, the 120-wan enumerated best plan is:

- C small, E small, G small, I small, J large;
- build cost 117 wan;
- spatial coverage 100%;
- capacity fulfillment 0.849858;
- effective service rate 0.743885;
- served weighted satisfaction 0.875305.

For comparison:

- C small, D large, G large: effective service rate 0.733559;
- B large, D small, F medium, J small: effective service rate 0.729990.

For the 140-wan scenario, the capacity-hard enumerated best plan is:

- A small, B large, G large, I medium;
- capacity fulfillment 1.000000;
- effective service rate 0.906503.

See `outputs/capacity_optimized_fair/decision_report.md` for the Chinese decision note.
