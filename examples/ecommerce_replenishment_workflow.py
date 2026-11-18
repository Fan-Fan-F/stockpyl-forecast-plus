#!/usr/bin/env python3
# ===============================================================================
# stockpyl - End-to-End E-commerce Demand Forecasting & Inventory Optimization
# ===============================================================================

"""
This example script demonstrates a complete end-to-end replenishment workflow:
1. Ingesting transactional e-commerce order logs.
2. Resampling & zero-filling demand time series per SKU.
3. Benchmarking multiple time-series forecasting models (SES, Holt-Winters, Croston, Auto).
4. Deriving safety stock, reorder point (r), base-stock (S), and EOQ (Q).
5. Building a two-echelon distribution network (Central CDC -> Regional FC).
6. Simulating the network under empirical customer demand and evaluating KPI metrics.
"""

import os
import sys

# Ensure src is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_dir not in sys.path:
	sys.path.insert(0, src_dir)

import numpy as np
from stockpyl.data_io import (
	load_sample_ecommerce_orders, aggregate_orders,
	orders_to_demand_source, build_ecommerce_fulfillment_network
)
from stockpyl.forecast import (
	simple_moving_average, simple_exponential_smoothing,
	holt_winters, croston, sba, auto_forecast,
	forecast_to_inventory_params
)
from stockpyl.sim import simulation


def run_ecommerce_replenishment_pipeline():
	print("=" * 80)
	print(" STOCKPYL E-COMMERCE DEMAND FORECASTING & REPLENISHMENT WORKFLOW")
	print("=" * 80)

	# 1. Ingest order transactions
	print("\n[Step 1] Ingesting Transactional Order History...")
	orders = load_sample_ecommerce_orders()
	print(f"  Successfully loaded {len(orders)} order records.")

	# 2. Daily aggregation per SKU
	print("\n[Step 2] Resampling into Continuous Daily Demand (Zero-Filled)...")
	daily_demand = aggregate_orders(orders, freq='D')
	for sku, ts in daily_demand.items():
		zero_days = np.sum(ts == 0)
		print(f"  SKU: {sku:15s} | {len(ts)} days | Total Qty: {np.sum(ts):6.0f} | Mean/Day: {np.mean(ts):5.1f} | Zero-Demand Days: {zero_days:2d}")

	# 3. Model benchmarking on fast-moving SKU
	fast_sku = 'SKU-FAST-101'
	ts_fast = daily_demand[fast_sku]
	print(f"\n[Step 3] Forecasting Benchmark for {fast_sku}...")

	models = {
		'SMA-7': simple_moving_average(ts_fast, window=7, forecast_periods=7),
		'SES': simple_exponential_smoothing(ts_fast, forecast_periods=7),
		'Holt-Winters (7d)': holt_winters(ts_fast, seasonal_periods=7, forecast_periods=7),
		'Auto-Forecast': auto_forecast(ts_fast, forecast_periods=7)
	}

	print(f"  {'Model':20s} | {'MAE':8s} | {'RMSE':8s} | {'WAPE':8s} | {'Next-7d Mean':12s}")
	print("  " + "-" * 65)
	for name, res in models.items():
		m = res.metrics
		print(f"  {name:20s} | {m.get('mae', 0):8.2f} | {m.get('rmse', 0):8.2f} | {m.get('wape', 0)*100:7.2f}% | {res.mean_forecast:12.2f}")

	# 4. Intermittent Demand Benchmark for Slow Mover
	slow_sku = 'SKU-SLOW-303'
	ts_slow = daily_demand[slow_sku]
	print(f"\n[Step 4] Intermittent Demand Forecasting for Slow-Mover ({slow_sku})...")
	res_croston = croston(ts_slow, alpha=0.1, forecast_periods=7)
	res_sba = sba(ts_slow, alpha=0.1, forecast_periods=7)
	print(f"  Croston Rate: {res_croston.forecasts[0]:.3f} units/day (WAPE: {res_croston.metrics['wape']*100:.2f}%)")
	print(f"  SBA Rate:     {res_sba.forecasts[0]:.3f} units/day (WAPE: {res_sba.metrics['wape']*100:.2f}%)")

	# 5. Inventory Parameter Derivation (Forecast-to-Inventory Bridge)
	best_fast = models['Auto-Forecast']
	inv_params = forecast_to_inventory_params(
		mean_demand=best_fast.mean_forecast,
		residual_std=best_fast.residual_std,
		lead_time=2.0,            # 2 days replenishment lead time
		service_level=0.95,       # 95% cycle service level
		holding_cost=1.5,         # $1.50/unit/day
		fixed_cost=40.0           # $40 order placement setup cost
	)
	print("\n[Step 5] Derived Replenishment Parameters (95% SL):")
	print(f"  Daily Demand Mean:         {inv_params.mean_demand_per_period:.2f} units/day")
	print(f"  Lead Time Demand (LTD):    {inv_params.lead_time_demand_mean:.2f} units (std: {inv_params.lead_time_demand_std:.2f})")
	print(f"  Safety Stock (SS):         {inv_params.safety_stock:.2f} units")
	print(f"  Reorder Point (r):         {inv_params.reorder_point:.2f} units")
	print(f"  Base Stock Level (S):      {inv_params.base_stock_level:.2f} units")
	print(f"  Economic Order Qty (EOQ):  {inv_params.economic_order_quantity:.2f} units")

	# 6. Multi-Echelon Network Assembly & Simulation
	print("\n[Step 6] Simulating 2-Echelon Network (CDC -> FC) over 30 Days...")
	network = build_ecommerce_fulfillment_network(
		sku_id=fast_sku,
		daily_demand_series=ts_fast,
		service_level=0.95,
		supplier_lead_time=2,
		fulfillment_lead_time=1,
		demand_mode='TRACE'  # Replay empirical customer orders
	)

	total_cost = simulation(network, num_periods=30, progress_bar=False)
	fc_node = network.nodes_by_index[2]

	total_demand = sum(fc_node.state_vars[t].inbound_order[None][fc_node.product_indices[0]] for t in range(30))
	total_met = fc_node.state_vars[29].get_demand_met_from_stock_cumul()
	fill_rate = (total_met / total_demand * 100) if total_demand > 0 else 100.0
	holding_cost = sum(fc_node.state_vars[t].holding_cost_incurred for t in range(30))
	stockout_cost = sum(fc_node.state_vars[t].stockout_cost_incurred for t in range(30))

	print("\n[Simulation Performance Results]")
	print(f"  Total Simulated Periods:   30 Days")
	print(f"  Total Units Demanded:      {total_demand:.0f}")
	print(f"  Total Units Met from Stock:{total_met:.0f}")
	print(f"  Fulfillment Fill Rate:     {fill_rate:.2f}%")
	print(f"  Total Incurred Cost:       ${total_cost:,.2f}")
	print(f"    - Holding Cost:          ${holding_cost:,.2f}")
	print(f"    - Stockout Penalty:      ${stockout_cost:,.2f}")
	print("\nWorkflow completed successfully!")
	print("=" * 80)


if __name__ == '__main__':
	run_ecommerce_replenishment_pipeline()
