# stockpyl-forecast-plus

> **面向电商全渠道履约场景的多级库存优化、需求预测与仿真引擎**
> 
> *A Python library for multi-echelon inventory optimization, e-commerce demand forecasting, and supply chain simulation.*

---

## 中文说明

### 简介

`stockpyl-forecast-plus` 是在经典多级库存优化与仿真算法库 `stockpyl` 基础上的二次开发增强版。针对电商全渠道履约中的**长尾品高零频、季节爆款尖峰、订单流水非规则时间戳以及跨级履约网络补货联动**等痛点，本项目补齐了端到端电商需求预测引擎、订单数据导入聚合接口、经验订单轨迹重放需求源，并彻底修复了原仓库核心仿真成本统计与网络拓扑初始化的严重 Bug。

### 功能与改进点（对应 CHANGES.md）

1. **[Bug Fix] 修复仿真成本核算与网络容量初始化缺陷**
   - 修复了原仓库在 `sim.py` 中直接调用不存在的 `n.fixed_cost` 导致的所有仿真成本崩溃（消除 `AttributeError: 'SupplyChainNode' object has no attribute 'fixed_cost'`）；
   - 在 `SupplyChainNode`、`SupplyChainProduct` 默认属性集中规范化挂载 `fixed_cost`，并在 `NodeStateVars` 中健全 `fixed_cost_incurred` 的记录、序列化与向下兼容；
   - 重构 `network_from_edges` 中对 `inventory_capacity` 及 `inventory_capacity_type` 的解析与初始化逻辑，修复原版无容量节点被强制实例化为 `InventoryCapacity` 且污染中断过程 `dp` 的缺陷；
   - 清理测试套件中残留的未定义变量碎片，原项目 95 个失败测试用例全部修复清零，实现全套件 100% 通过（763 passed）。

2. **[新功能] 电商订单历史需求预测引擎（`stockpyl.forecast`）**
   - **多模型预测体系**：提供移动平均（SMA / WMA）、单指数平滑（SES，支持网格/L-BFGS-B 参数自适应寻优）、Holt 双参数线性趋势、Holt-Winters 季节性平滑（加法与乘法季节性）；
   - **电商长尾与慢动品预测**：针对大量零销量期的长尾配件/慢动 SKU，实现 Croston 间歇性需求分解模型与 Syntetos-Boylan Approximation (SBA) 无偏修正模型；
   - **预测精度度量体系**：提供 MAE、RMSE、MAPE、WAPE（电商非零分母加权误差率，有效规避零除问题）、SMAPE 综合评测接口；
   - **预测到库存策略自动推导（Forecast-to-Inventory Bridge）**：基于预测期望值 $\hat{\mu}$、残差方差 $\hat{\sigma}_e$、提前期 $L$ 与目标服务水平 $SL$，一键推导安全库存（Safety Stock）、再订货点（ROP $r$）、基本库存目标（Base Stock $S$）及经济订货批量（EOQ $Q$），并可直接实例化为 stockpyl 的 `Policy` 对象。

3. **[新功能] 电商订单流水数据导入与聚合引擎（`stockpyl.data_io`）**
   - **全格式订单流水读取**：`load_order_history()` 支持 CSV 文件、Pandas DataFrame、字典列表，自动解析多种日期时间格式，支持时间戳、SKU ID、订货量、履约仓等字段自定义映射；
   - **连续时间序列聚合与零需求补齐**：`aggregate_orders()` 支持按日（D）、按周（W）进行时间桶聚合重采样，自动对未发生销售的日期补零，生成标准连续时间序列；
   - **订单驱动需求源转换**：`orders_to_demand_source()` 可将订单序列自动拟合为正态分布（Normal）、泊松分布（Poisson）或经验轨迹重放模式（TRACE）；
   - **电商多级履约网络快速组装**：`build_ecommerce_fulfillment_network()` 便捷装配「外部供应商 -> 中心仓 (CDC) -> 区域分发中心/前置仓 (FC)」多级拓扑。

4. **[优化与增强] 订单真实轨迹驱动需求源（Trace-Driven Demand Replay）**
   - 在 `DemandSource` 中拓展 `type='TRACE'` 模式，支持按电商真实历史订单序列逐期回放，无需强加正态分布或泊松分布的理论假设，支持电商业务的精细化离线回测（Backtesting）。

5. **[内置样例与端到端实战] 电商订单样例数据集与补货工作流**
   - 内置包含 813 条真实电商交易记录的样例数据集 `src/stockpyl/datasets/ecommerce_orders.csv`，涵盖高频快消（SKU-FAST-101）、季节爆款（SKU-SEAS-202）及长尾慢动备件（SKU-SLOW-303）；
   - 随包提供端到端实战脚本 `examples/ecommerce_replenishment_workflow.py`，完整串联订单导入、预测比选、策略计算、网络组装与仿真回测。

---

### 快速开始

#### 1. 环境准备与安装

```bash
git clone https://github.com/Fan-Fan-F/stockpyl-forecast-plus.git
cd stockpyl-forecast-plus

# 安装依赖与项目（推荐可编辑模式）
pip install -e .
```

#### 2. 运行电商补货全流程实战示例

运行内置的端到端实战工作流（从订单读取、预测比选到多级网络仿真）：

```bash
python3 examples/ecommerce_replenishment_workflow.py
```

#### 3. 核心 API 快速上手

**（1）加载电商历史订单并聚合：**
```python
from stockpyl.data_io import load_sample_ecommerce_orders, aggregate_orders

# 载入内置订单数据集
orders = load_sample_ecommerce_orders()

# 按日聚合各 SKU 连续时间序列（自动补 0）
daily_demand = aggregate_orders(orders, freq='D')
ts_fast = daily_demand['SKU-FAST-101']
```

**（2）多模型预测与评估：**
```python
from stockpyl.forecast import simple_exponential_smoothing, holt_winters, auto_forecast

# 自动比选最优预测模型（以 WAPE 为准）
best_model = auto_forecast(ts_fast, forecast_periods=7)
print("最优模型:", best_model.method)
print("未来7天日均预测量:", best_model.mean_forecast)
print("历史预测 WAPE:", best_model.metrics['wape'])
```

**（3）自动推导安全库存与再订货点（ROP）：**
```python
from stockpyl.forecast import forecast_to_inventory_params

inv = forecast_to_inventory_params(
    mean_demand=best_model.mean_forecast,
    residual_std=best_model.residual_std,
    lead_time=2.0,       # 提前期 2 天
    service_level=0.95,  # 目标服务水平 95%
    holding_cost=1.5,    # 日持有成本 $1.5
    fixed_cost=40.0      # 订购固定成本 $40
)

print(f"安全库存 SS: {inv.safety_stock:.2f}")
print(f"再订货点 r:   {inv.reorder_point:.2f}")
print(f"经济批量 EOQ: {inv.economic_order_quantity:.2f}")

# 直接转化为 stockpyl Policy 对象
policy_bs = inv.to_policy(policy_type='BS')
```

**（4）组装多级网络并仿真评估：**
```python
from stockpyl.data_io import build_ecommerce_fulfillment_network
from stockpyl.sim import simulation

# 构建 CDC -> FC 履约网络并绑定实证需求轨迹
network = build_ecommerce_fulfillment_network(
    sku_id='SKU-FAST-101',
    daily_demand_series=ts_fast,
    service_level=0.95,
    demand_mode='TRACE'
)

# 仿真 30 天运作
total_cost = simulation(network, num_periods=30, progress_bar=False)
fc = network.nodes_by_index[2]
print(f"30 天总运营成本: ${total_cost:,.2f}")
print(f"履约满足率: {fc.state_vars[29].get_demand_met_from_stock_cumul() / sum(fc.state_vars[t].inbound_order[None][fc.product_indices[0]] for t in range(30)):.2%}")
```

#### 4. 运行单元测试

```bash
# 运行全部 773 个单元测试
pytest -q tests/
```

---

## English Documentation

### Introduction

`stockpyl-forecast-plus` is an enhanced distribution of the [stockpyl](https://github.com/LarrySnyder/stockpyl) library for multi-echelon inventory optimization and simulation. Tailored for modern omnichannel e-commerce fulfillment, this project introduces transactional order ingestion, automated time-series demand forecasting (including seasonal trends and intermittent demand for long-tail items), empirical trace-driven demand replay, and fixes core simulation cost-accounting bugs present in the upstream repository.

### Key Features and Improvements (Corresponding to CHANGES.md)

1. **[Bug Fix] Simulation Cost Calculation & Capacity Initialization Fixes**
   - Resolved the upstream `AttributeError: 'SupplyChainNode' object has no attribute 'fixed_cost'` by safely accessing node attributes in `sim.py`.
   - Standardized `fixed_cost` across `SupplyChainNode` and `SupplyChainProduct` default attribute registries and enabled backward-compatible serialization in `NodeStateVars`.
   - Fixed erroneous unconditional initialization of `InventoryCapacity` in `network_from_edges` that corrupted `dp` (`DisruptionProcess`) and broke node equality (`deep_equal_to`).
   - Cleaned up dangling test fixtures; brought the entire test suite to 100% pass status (763 passed, 10 skipped, 0 failed).

2. **[New Module] E-commerce Demand Forecasting Engine (`stockpyl.forecast`)**
   - **Time Series Models**: Simple Moving Average (SMA), Weighted Moving Average (WMA), Simple Exponential Smoothing (SES with L-BFGS-B parameter tuning), Holt's Linear Trend, and Holt-Winters Additive/Multiplicative Seasonality.
   - **Intermittent & Slow-Moving SKUs**: Croston's decomposition method and Syntetos-Boylan Approximation (SBA) for sparse, zero-heavy demand streams.
   - **Accuracy Benchmarking**: MAE, RMSE, MAPE, WAPE (zero-safe weighted error), and SMAPE metrics.
   - **Forecast-to-Inventory Bridge**: Automatically derives Safety Stock (SS), Reorder Point ($r$), Base-Stock level ($S$), and Economic Order Quantity ($Q$) from forecast expectations and residual variance for seamless policy instantiation.

3. **[New Module] Order Data Ingestion & Continuous Aggregation (`stockpyl.data_io`)**
   - Ingests raw transactional logs from CSV files, DataFrames, or dictionary records.
   - Resamples timestamps into daily or weekly buckets, padding missing sales days with zero demand.
   - Automatically converts order sequences into stockpyl `DemandSource` instances (Normal, Poisson, or Trace).
   - Rapid assembly helper `build_ecommerce_fulfillment_network()` for multi-tier fulfillment pipelines (Supplier -> Central Distribution Center -> Regional Fulfillment Centers).

4. **[Optimization & Enhancement] Trace-Driven Empirical Demand Replay**
   - Extended `DemandSource` with `type='TRACE'` to replay real historical customer orders without parametric distribution assumptions, supporting realistic backtesting.

5. **[Sample Dataset & End-to-End Workflow]**
   - Included `src/stockpyl/datasets/ecommerce_orders.csv` containing 813 real-world multi-SKU transactions across 2 fulfillment hubs.
   - Shipped `examples/ecommerce_replenishment_workflow.py` executing an end-to-end pipeline in ~2 seconds.

---

### Quick Start

#### Installation

```bash
git clone https://github.com/Fan-Fan-F/stockpyl-forecast-plus.git
cd stockpyl-forecast-plus
pip install -e .
```

#### Run the End-to-End E-Commerce Replenishment Example

```bash
python3 examples/ecommerce_replenishment_workflow.py
```

#### Run Tests

```bash
pytest -q tests/
```

---

## 来源与致谢 / Provenance & Acknowledgements

本项目基于 [stockpyl](https://github.com/LarrySnyder/stockpyl) 二次开发，原项目由 LarrySnyder 以 MIT 许可证开源。

*This project is a secondary development based on [stockpyl](https://github.com/LarrySnyder/stockpyl), originally developed by LarrySnyder and open-sourced under the MIT License.*

---

## 许可证 / License

本项目沿用原项目的 MIT 开源许可证。全文见根目录 [LICENSE](LICENSE)。

*This project continues to be licensed under the MIT License. See [LICENSE](LICENSE) for full terms.*
