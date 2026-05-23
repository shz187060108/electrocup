# 电工杯 B 题：嵌入式社区养老服务站优化项目

本项目为“完整规划版”代码包，按机器学习项目风格组织目录，但模型本身采用数学规划、枚举搜索和固定点迭代方法。

## 目录结构

```text
configs/default.yaml          # 统一配置文件
data/raw/                     # 原始附件数据
experiments/                  # 四个问题与一键运行入口
src/data/                     # 数据读取模块
src/models/                   # 四个问题对应模型
src/solvers/                  # 固定点迭代求解器
src/utils/                    # 工具函数
outputs/results/              # 运行后结果表输出
outputs/figures/              # 运行后图片输出
```

## 四个问题对应入口

```bash
python experiments/run_problem1.py
python experiments/run_problem2.py
python experiments/run_problem3.py
python experiments/run_problem4.py
```

一键运行全部问题：

```bash
python experiments/run_all.py
```

## 当前版本的主要模型

- 问题一：多状态 Markov 人口预测与消费约束需求修正模型
- 问题二：基于 CMCLP 与双层规划思想的服务站选址容量优化模型
- 问题三：补贴约束下的临界点定价与潜在需求释放模型
- 问题四：基于情景分析与后悔值的鲁棒性评价模型

## 完整规划版说明

本版本不再采用快速预筛选或局部重优化：

1. 问题二会枚举所有预算可行的位置和规模组合；
2. 问题三会对五类收费服务进行临界价格全组合枚举；
3. 问题四会对每个情景重新运行问题一、问题二和问题三；
4. 满意度掉档损失拆分为距离、响应、价格和综合满意度四类。

完整运行会比上一版慢，但小区数量只有 10 个，搜索规模仍然可控。

## 依赖安装

```bash
pip install -r requirements.txt
```

## 输出文件

运行后主要输出包括：

- `problem1_year5_population.csv`
- `problem1_theoretical_monthly_demand.csv`
- `problem1_budgeted_monthly_demand.csv`
- `problem2_top500_plan_summary.csv`
- `problem2_best_assignment.csv`
- `problem2_best_station_status.csv`
- `problem3_station_prices.csv`
- `problem3_station_finance.csv`
- `problem3_latent_demand_release.csv`
- `problem4_scenario_summary.csv`
- `all_results_summary.xlsx`
