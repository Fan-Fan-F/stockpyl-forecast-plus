# ===============================================================================
# stockpyl - Unit Tests for Demand Forecasting Module
# ===============================================================================

import unittest
import numpy as np

from stockpyl.forecast import (
	mae, rmse, wape, mape, smape, evaluate_forecast,
	simple_moving_average, weighted_moving_average,
	simple_exponential_smoothing, holt_linear, holt_winters,
	croston, sba, auto_forecast,
	forecast_to_inventory_params
)
from stockpyl.policy import Policy


class TestForecastMetrics(unittest.TestCase):
	"""Test forecast accuracy metric calculations."""

	def test_perfect_forecast(self):
		y = [10.0, 20.0, 30.0]
		self.assertEqual(mae(y, y), 0.0)
		self.assertEqual(rmse(y, y), 0.0)
		self.assertEqual(wape(y, y), 0.0)
		self.assertEqual(mape(y, y), 0.0)
		self.assertEqual(smape(y, y), 0.0)

	def test_known_errors(self):
		y_true = [100.0, 200.0]
		y_pred = [110.0, 190.0]
		self.assertAlmostEqual(mae(y_true, y_pred), 10.0)
		self.assertAlmostEqual(rmse(y_true, y_pred), 10.0)
		# WAPE: (10 + 10) / (100 + 200) = 20 / 300 = 0.066667
		self.assertAlmostEqual(wape(y_true, y_pred), 20.0 / 300.0, places=5)
		# MAPE: (0.1 + 0.05) / 2 = 0.075
		self.assertAlmostEqual(mape(y_true, y_pred), 0.075, places=5)

	def test_wape_zero_actuals(self):
		y_true = [0.0, 0.0]
		y_pred = [5.0, 5.0]
		self.assertEqual(wape(y_true, y_pred), 1.0)
		self.assertEqual(wape(y_true, [0.0, 0.0]), 0.0)


class TestTimeSeriesForecasting(unittest.TestCase):
	"""Test time series forecasting methods."""

	def test_sma(self):
		history = [10, 20, 35, 40, 55]
		res = simple_moving_average(history, window=3, forecast_periods=2)
		self.assertEqual(res.method, 'SMA')
		self.assertEqual(len(res.forecasts), 2)
		# Window of 3 on [35, 40, 55] -> (35+40+55)/3 = 130/3 = 43.3333
		self.assertAlmostEqual(res.forecasts[0], 130.0 / 3.0, places=4)
		self.assertGreater(res.residual_std, 0.0)

	def test_wma(self):
		history = [10, 20, 30]
		res = weighted_moving_average(history, weights=[1, 2, 3], forecast_periods=1)
		# (10*1 + 20*2 + 30*3) / (1+2+3) = (10 + 40 + 90)/6 = 140/6 = 23.3333
		self.assertAlmostEqual(res.forecasts[0], 140.0 / 6.0, places=4)

	def test_ses(self):
		np.random.seed(42)
		history = 50.0 + np.random.normal(0, 3, size=30)
		res = simple_exponential_smoothing(history, alpha=0.3, forecast_periods=3)
		self.assertEqual(res.method, 'SES')
		self.assertEqual(len(res.forecasts), 3)
		self.assertTrue(40.0 < res.forecasts[0] < 60.0)

		# Auto-fit alpha
		res_auto = simple_exponential_smoothing(history, alpha=None, forecast_periods=1)
		self.assertIn('alpha', res_auto.parameters)
		self.assertTrue(0.01 <= res_auto.parameters['alpha'] <= 0.99)

	def test_holt_linear_trend(self):
		# Upward trend: 10, 15, 20, 25, 30, 35, 40
		history = np.arange(10, 60, 5, dtype=float)
		res = holt_linear(history, alpha=0.5, beta=0.2, forecast_periods=3)
		self.assertEqual(res.method, 'HoltLinear')
		self.assertEqual(len(res.forecasts), 3)
		# Forecast should continue upward
		self.assertGreater(res.forecasts[1], res.forecasts[0])
		self.assertGreater(res.forecasts[2], res.forecasts[1])

	def test_holt_winters(self):
		# Weekly seasonal pattern with 7 periods
		weekly_pattern = np.array([20, 25, 22, 30, 50, 70, 35], dtype=float)
		history = np.tile(weekly_pattern, 4)  # 4 weeks = 28 periods
		res = holt_winters(history, seasonal_periods=7, forecast_periods=7, seasonal_type='add')
		self.assertEqual(res.method, 'HoltWinters')
		self.assertEqual(len(res.forecasts), 7)
		# Saturday should be peak in next cycle
		self.assertEqual(np.argmax(res.forecasts), 5)


class TestIntermittentForecasting(unittest.TestCase):
	"""Test intermittent demand forecasting for slow movers (Croston and SBA)."""

	def test_croston_and_sba(self):
		# Intermittent pattern: orders of size ~10 occurring every 3-4 periods
		history = [0, 10, 0, 0, 12, 0, 0, 0, 10, 0, 8, 0, 0, 11]
		res_croston = croston(history, alpha=0.1, forecast_periods=3)
		res_sba = sba(history, alpha=0.1, forecast_periods=3)

		self.assertEqual(res_croston.method, 'Croston')
		self.assertEqual(res_sba.method, 'SBA')
		# Forecasts should be positive non-zero
		self.assertGreater(res_croston.forecasts[0], 0.0)
		self.assertGreater(res_sba.forecasts[0], 0.0)
		# SBA should deflate Croston by (1 - alpha/2) = 0.95
		self.assertAlmostEqual(res_sba.forecasts[0], res_croston.forecasts[0] * 0.95, places=5)

	def test_auto_forecast_selection(self):
		# High intermittency triggers SBA/Croston
		intermittent = [0, 0, 5, 0, 0, 0, 6, 0, 0, 4, 0, 0, 0, 5]
		best_int = auto_forecast(intermittent, forecast_periods=2)
		self.assertIn(best_int.method, ('SBA', 'Croston', 'SES'))

		# Steady demand triggers standard model
		steady = np.array([50, 52, 49, 51, 50, 48, 53, 50, 51, 49], dtype=float)
		best_std = auto_forecast(steady, forecast_periods=2)
		self.assertIn(best_std.method, ('SMA', 'SES', 'HoltLinear'))


class TestForecastToInventoryBridge(unittest.TestCase):
	"""Test conversion from forecast outputs to stockpyl Policy parameters."""

	def test_inventory_parameter_derivation(self):
		# Mean demand = 100/day, error std = 15/day, lead time = 4 days, 95% service level
		inv_params = forecast_to_inventory_params(
			mean_demand=100.0,
			residual_std=15.0,
			lead_time=4.0,
			service_level=0.95,
			holding_cost=2.0,
			fixed_cost=50.0
		)

		# Lead time demand mean: 100 * 4 = 400
		self.assertEqual(inv_params.lead_time_demand_mean, 400.0)
		# Lead time demand std: 15 * sqrt(4) = 30
		self.assertEqual(inv_params.lead_time_demand_std, 30.0)
		# z for 0.95 ~ 1.64485
		self.assertAlmostEqual(inv_params.safety_factor_z, 1.64485, places=4)
		# Safety stock: 1.64485 * 30 ~ 49.3456
		self.assertAlmostEqual(inv_params.safety_stock, 1.64485 * 30.0, places=3)
		# ROP = 400 + 49.3456
		self.assertAlmostEqual(inv_params.reorder_point, 400.0 + inv_params.safety_stock)
		# EOQ: sqrt(2 * 100 * 50 / 2) = sqrt(5000) ~ 70.7107
		self.assertAlmostEqual(inv_params.economic_order_quantity, np.sqrt(5000.0), places=4)

		# Test policy instantiation
		bs_policy = inv_params.to_policy('BS')
		self.assertIsInstance(bs_policy, Policy)
		self.assertEqual(bs_policy.type, 'BS')
		self.assertAlmostEqual(bs_policy.base_stock_level, inv_params.base_stock_level)

		rq_policy = inv_params.to_policy('rQ')
		self.assertEqual(rq_policy.type, 'rQ')
		self.assertAlmostEqual(rq_policy.reorder_point, inv_params.reorder_point)
		self.assertAlmostEqual(rq_policy.order_quantity, inv_params.economic_order_quantity)


if __name__ == '__main__':
	unittest.main()
